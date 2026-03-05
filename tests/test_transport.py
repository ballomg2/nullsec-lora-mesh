"""
Tests for transport layer — reliable delivery, fragmentation, dedup.
"""

import pytest
import struct
import time

from transport.reliable import (
    TransportLayer,
    TransportSegment,
    TransportStats,
    ReceivedData,
    SegmentType,
    DeliveryStatus,
    FragmentBuffer,
    MAX_SEGMENT_SIZE,
)


class TestTransportSegment:
    """Test transport segment encoding/decoding."""

    def test_encode_decode_data(self):
        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=42,
            msg_id=1,
            payload=b"Hello transport!",
        )
        encoded = seg.encode()
        decoded = TransportSegment.decode(encoded)

        assert decoded.seg_type == SegmentType.DATA
        assert decoded.sequence == 42
        assert decoded.msg_id == 1
        assert decoded.payload == b"Hello transport!"

    def test_encode_decode_ack(self):
        seg = TransportSegment(
            seg_type=SegmentType.ACK,
            sequence=1,
            ack_number=42,
        )
        encoded = seg.encode()
        decoded = TransportSegment.decode(encoded)

        assert decoded.seg_type == SegmentType.ACK
        assert decoded.ack_number == 42

    def test_checksum_verification(self):
        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=1,
            payload=b"test data",
        )
        encoded = seg.encode()

        # Corrupt a byte in the payload
        corrupted = bytearray(encoded)
        corrupted[-1] ^= 0xFF
        corrupted = bytes(corrupted)

        with pytest.raises(ValueError, match="Checksum mismatch"):
            TransportSegment.decode(corrupted)

    def test_empty_payload(self):
        seg = TransportSegment(
            seg_type=SegmentType.PING,
            sequence=1,
            payload=b"",
        )
        encoded = seg.encode()
        decoded = TransportSegment.decode(encoded)
        assert decoded.payload == b""

    def test_header_size(self):
        assert TransportSegment.HEADER_SIZE == 12

    def test_total_size(self):
        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=1,
            payload=b"12345",
        )
        assert seg.total_size == 12 + 5

    def test_sequence_wraps(self):
        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=0xFFFF,
            payload=b"wrap",
        )
        encoded = seg.encode()
        decoded = TransportSegment.decode(encoded)
        assert decoded.sequence == 0xFFFF

    def test_decode_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            TransportSegment.decode(b"short")

    def test_fragment_fields(self):
        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=10,
            frag_id=100,
            frag_total=5,
            msg_id=3,
            payload=b"fragment",
        )
        encoded = seg.encode()
        decoded = TransportSegment.decode(encoded)

        assert decoded.frag_id == 100
        assert decoded.frag_total == 5
        assert decoded.msg_id == 3

    def test_crc16_deterministic(self):
        """Same data should always produce same checksum."""
        data = b"deterministic test"
        crc1 = TransportSegment._compute_checksum(data)
        crc2 = TransportSegment._compute_checksum(data)
        assert crc1 == crc2

    def test_crc16_different_data(self):
        """Different data should produce different checksums."""
        crc1 = TransportSegment._compute_checksum(b"data1")
        crc2 = TransportSegment._compute_checksum(b"data2")
        assert crc1 != crc2


class TestFragmentBuffer:
    """Test fragment reassembly buffer."""

    def test_incomplete(self):
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.segments[0] = b"aaa"
        buf.segments[1] = b"bbb"
        assert not buf.is_complete

    def test_complete(self):
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.segments[0] = b"aaa"
        buf.segments[1] = b"bbb"
        buf.segments[2] = b"ccc"
        assert buf.is_complete

    def test_reassemble(self):
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.segments[0] = b"Hello "
        buf.segments[1] = b"mesh "
        buf.segments[2] = b"world!"
        assert buf.reassemble() == b"Hello mesh world!"

    def test_reassemble_order(self):
        """Fragments should be reassembled in order regardless of arrival."""
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.segments[2] = b"C"
        buf.segments[0] = b"A"
        buf.segments[1] = b"B"
        assert buf.reassemble() == b"ABC"

    def test_reassemble_incomplete_raises(self):
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.segments[0] = b"only one"
        with pytest.raises(ValueError):
            buf.reassemble()

    def test_expiry(self):
        buf = FragmentBuffer(frag_id=1, total=3, src=0x01)
        buf.created = time.time() - 120  # 2 minutes ago
        assert buf.is_expired


class TestTransportLayer:
    """Test the transport layer."""

    def test_init(self):
        tl = TransportLayer(node_id=0x01)
        assert tl.node_id == 0x01
        assert tl.stats.messages_sent == 0

    def test_send_small_message(self):
        sent_segments = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: sent_segments.append((dest, data)))

        msg_id = tl.send(dest=0x02, data=b"Hello!")
        assert msg_id > 0
        assert len(sent_segments) == 1
        assert sent_segments[0][0] == 0x02
        assert tl.stats.messages_sent == 1

    def test_send_large_message_fragments(self):
        """Messages larger than MAX_SEGMENT_SIZE should be fragmented."""
        sent_segments = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: sent_segments.append((dest, data)))

        large_data = b"X" * (MAX_SEGMENT_SIZE * 3)
        tl.send(dest=0x02, data=large_data)

        # Should have sent multiple segments
        assert len(sent_segments) > 1
        assert tl.stats.segments_sent > 1

    def test_receive_and_ack(self):
        """Processing a DATA segment should produce an ACK."""
        acks_sent = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: acks_sent.append((dest, data)))

        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=100,
            msg_id=1,
            payload=b"incoming",
        )
        tl.process_segment(src=0x02, segment_data=seg.encode())

        # Should have sent an ACK
        assert len(acks_sent) == 1
        ack = TransportSegment.decode(acks_sent[0][1])
        assert ack.seg_type == SegmentType.ACK
        assert ack.ack_number == 100

    def test_receive_delivers_message(self):
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: None)  # Swallow ACKs

        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=1,
            msg_id=1,
            payload=b"test message",
        )
        tl.process_segment(src=0x02, segment_data=seg.encode())

        messages = tl.receive()
        assert len(messages) == 1
        assert messages[0].data == b"test message"
        assert messages[0].src == 0x02

    def test_duplicate_detection(self):
        """Same segment received twice should only deliver once."""
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: None)

        seg = TransportSegment(
            seg_type=SegmentType.DATA,
            sequence=50,
            msg_id=5,
            payload=b"duplicate",
        )
        encoded = seg.encode()

        tl.process_segment(src=0x02, segment_data=encoded)
        tl.process_segment(src=0x02, segment_data=encoded)

        messages = tl.receive()
        assert len(messages) == 1
        assert tl.stats.duplicates_detected == 1

    def test_ack_marks_delivered(self):
        sent = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: sent.append((dest, data)))

        msg_id = tl.send(dest=0x02, data=b"hello")

        # Get the sequence number of the sent segment
        data_seg = TransportSegment.decode(sent[0][1])

        # Simulate receiving ACK
        ack = TransportSegment(
            seg_type=SegmentType.ACK,
            sequence=999,
            ack_number=data_seg.sequence,
        )
        tl.process_segment(src=0x02, segment_data=ack.encode())

        assert tl.stats.acks_received == 1

    def test_ping_pong(self):
        """PING should trigger PONG response."""
        responses = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: responses.append((dest, data)))

        ping = TransportSegment(
            seg_type=SegmentType.PING,
            sequence=1,
            payload=struct.pack(">d", time.time()),
        )
        tl.process_segment(src=0x02, segment_data=ping.encode())

        assert len(responses) == 1
        pong = TransportSegment.decode(responses[0][1])
        assert pong.seg_type == SegmentType.PONG

    def test_rtt_estimation(self):
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: None)

        # Simulate a PONG with timestamp from 100ms ago
        ts = time.time() - 0.1
        pong = TransportSegment(
            seg_type=SegmentType.PONG,
            sequence=1,
            payload=struct.pack(">d", ts),
        )
        tl.process_segment(src=0x02, segment_data=pong.encode())

        rtt = tl.get_rtt(0x02)
        assert rtt is not None
        assert rtt > 0.05  # At least ~100ms

    def test_tick_no_crash(self):
        """tick() should not crash with no pending messages."""
        tl = TransportLayer(node_id=0x01)
        tl.tick()  # Should not raise

    def test_invalid_segment_ignored(self):
        tl = TransportLayer(node_id=0x01)
        tl.process_segment(src=0x02, segment_data=b"garbage")
        assert tl.stats.checksum_errors == 1

    def test_delivery_callback(self):
        delivered = []
        sent = []
        tl = TransportLayer(node_id=0x01)
        tl.on_send(lambda dest, data: sent.append((dest, data)))

        msg_id = tl.send(
            dest=0x02, data=b"callback test",
            on_delivered=lambda: delivered.append(True),
        )

        # ACK it
        data_seg = TransportSegment.decode(sent[0][1])
        ack = TransportSegment(
            seg_type=SegmentType.ACK,
            sequence=999,
            ack_number=data_seg.sequence,
        )
        tl.process_segment(src=0x02, segment_data=ack.encode())

        assert len(delivered) == 1
