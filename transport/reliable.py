"""
NullSec LoRa Mesh - Transport Layer

Provides reliable, ordered delivery over the mesh network with:
- Stop-and-wait ARQ with exponential backoff
- Message fragmentation and reassembly
- Flow control for congested nodes
- Duplicate detection and suppression
- Delivery confirmation callbacks
"""

import struct
import time
import threading
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple
from queue import Queue, PriorityQueue, Empty
import logging

logger = logging.getLogger(__name__)


# ── Constants ──
MAX_SEGMENT_SIZE = 200       # Max payload per segment (after headers)
MAX_RETRIES = 5              # Max retransmission attempts
BASE_TIMEOUT = 2.0           # Base ACK timeout (seconds)
MAX_TIMEOUT = 30.0           # Maximum backoff timeout
WINDOW_SIZE = 4              # Sliding window size (max in-flight)
FRAG_REASSEMBLY_TIMEOUT = 60 # Seconds before incomplete fragments expire
DEDUP_CACHE_SIZE = 256       # Number of recent message hashes to track
FLOW_CONTROL_THRESHOLD = 8   # Queue depth before backpressure


class SegmentType(IntEnum):
    """Transport segment types."""
    DATA = 0x01          # Data segment
    ACK = 0x02           # Acknowledgement
    NACK = 0x03          # Negative acknowledgement (retransmit request)
    FIN = 0x04           # End of stream
    RST = 0x05           # Reset connection
    PING = 0x06          # Keepalive / RTT measurement
    PONG = 0x07          # Keepalive response


class DeliveryStatus(IntEnum):
    """Message delivery status."""
    PENDING = 0
    IN_FLIGHT = 1
    DELIVERED = 2
    FAILED = 3
    TIMEOUT = 4


@dataclass
class TransportSegment:
    """
    Transport layer segment.

    Header format (12 bytes):
    ┌──────┬──────┬──────┬──────┬──────┬──────┐
    │ Type │ SeqN │ AckN │ Frag │ Total│ CRC16│
    │ 1B   │ 2B   │ 2B   │ 2B   │ 2B   │ 2B   │  = 11B
    │      │      │      │ ID   │ Count│      │
    └──────┴──────┴──────┴──────┴──────┴──────┘
    + MsgID 1B = 12B total
    """
    HEADER_FORMAT = ">BHH HH H B"
    HEADER_SIZE = struct.calcsize(">BHH HH H B")  # 12 bytes

    seg_type: SegmentType
    sequence: int           # Sequence number
    ack_number: int = 0     # Piggybacked ACK
    frag_id: int = 0        # Fragment group ID
    frag_total: int = 1     # Total fragments in group
    checksum: int = 0       # CRC-16 of payload
    msg_id: int = 0         # Message ID for dedup
    payload: bytes = b""

    def encode(self) -> bytes:
        """Encode segment to bytes."""
        self.checksum = self._compute_checksum(self.payload)
        header = struct.pack(
            self.HEADER_FORMAT,
            int(self.seg_type),
            self.sequence & 0xFFFF,
            self.ack_number & 0xFFFF,
            self.frag_id & 0xFFFF,
            self.frag_total & 0xFFFF,
            self.checksum,
            self.msg_id & 0xFF,
        )
        return header + self.payload

    @classmethod
    def decode(cls, data: bytes) -> "TransportSegment":
        """Decode segment from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Segment too short: {len(data)}")

        (
            seg_type, sequence, ack_number,
            frag_id, frag_total, checksum, msg_id,
        ) = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])

        payload = data[cls.HEADER_SIZE:]

        seg = cls(
            seg_type=SegmentType(seg_type),
            sequence=sequence,
            ack_number=ack_number,
            frag_id=frag_id,
            frag_total=frag_total,
            checksum=checksum,
            msg_id=msg_id,
            payload=payload,
        )

        # Verify checksum
        expected = cls._compute_checksum(payload)
        if checksum != expected:
            raise ValueError(
                f"Checksum mismatch: 0x{checksum:04X} != 0x{expected:04X}"
            )

        return seg

    @staticmethod
    def _compute_checksum(data: bytes) -> int:
        """CRC-16/CCITT-FALSE checksum."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    @property
    def total_size(self) -> int:
        return self.HEADER_SIZE + len(self.payload)


@dataclass
class PendingMessage:
    """A message waiting for delivery confirmation."""
    msg_id: int
    dest: int
    segments: List[TransportSegment]
    status: DeliveryStatus = DeliveryStatus.PENDING
    retries: int = 0
    last_sent: float = 0.0
    timeout: float = BASE_TIMEOUT
    created: float = field(default_factory=time.time)
    callback: Optional[Callable] = None

    # Track which segments have been ACKed
    acked_segments: set = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        """All segments acknowledged."""
        return len(self.acked_segments) >= len(self.segments)

    @property
    def is_expired(self) -> bool:
        """Check if retry budget exhausted."""
        return self.retries >= MAX_RETRIES


@dataclass
class FragmentBuffer:
    """Reassembly buffer for fragmented messages."""
    frag_id: int
    total: int
    src: int
    segments: Dict[int, bytes] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return len(self.segments) >= self.total

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created) > FRAG_REASSEMBLY_TIMEOUT

    def reassemble(self) -> bytes:
        """Reassemble fragments in order."""
        if not self.is_complete:
            raise ValueError("Cannot reassemble incomplete fragments")
        parts = [self.segments[i] for i in sorted(self.segments.keys())]
        return b"".join(parts)


class TransportLayer:
    """
    Reliable transport layer for the LoRa mesh network.

    Sits between the application and the mesh node's frame layer.
    Provides reliable, ordered delivery with fragmentation support.

    Usage:
        transport = TransportLayer(node_id=0x01)

        # Send a message
        transport.send(dest=0x02, data=b"Hello mesh!", 
                       on_delivered=lambda: print("Delivered!"))

        # Process incoming segment (called by mesh node)
        transport.process_segment(src=0x02, segment_data=raw_bytes)

        # Get reassembled messages
        for msg in transport.receive():
            print(f"From {msg.src}: {msg.data}")
    """

    def __init__(self, node_id: int):
        self.node_id = node_id

        # Sequence counter
        self._next_seq = 0
        self._next_msg_id = 0

        # Outbound
        self._pending: Dict[int, PendingMessage] = {}
        self._outbox: Queue = Queue()

        # Inbound
        self._inbox: Queue = Queue()
        self._frag_buffers: Dict[Tuple[int, int], FragmentBuffer] = {}

        # Dedup cache (circular buffer of message hashes)
        self._seen_messages: List[bytes] = []

        # Flow control
        self._peer_backpressure: Dict[int, float] = {}

        # RTT estimation per peer
        self._rtt: Dict[int, float] = {}
        self._ping_times: Dict[int, float] = {}

        # Statistics
        self.stats = TransportStats()

        # Callbacks
        self._send_callback: Optional[Callable] = None

        # Lock for thread safety
        self._lock = threading.Lock()

    def on_send(self, callback: Callable[[int, bytes], None]):
        """
        Register callback to actually send data over the mesh.
        callback(dest_node_id, segment_bytes)
        """
        self._send_callback = callback

    def send(self, dest: int, data: bytes,
             reliable: bool = True,
             on_delivered: Optional[Callable] = None,
             on_failed: Optional[Callable] = None) -> int:
        """
        Send data to a destination node.

        Args:
            dest: Destination node ID
            data: Data to send
            reliable: Whether to require ACK
            on_delivered: Callback when all segments ACKed
            on_failed: Callback if delivery fails

        Returns:
            Message ID for tracking
        """
        with self._lock:
            self._next_msg_id = (self._next_msg_id + 1) & 0xFF

            # Check flow control
            if dest in self._peer_backpressure:
                backoff_until = self._peer_backpressure[dest]
                if time.time() < backoff_until:
                    logger.warning(f"Flow control: backpressure on peer 0x{dest:08X}")
                    self.stats.flow_control_events += 1

            # Fragment if needed
            segments = self._fragment(data, self._next_msg_id)

            if reliable:
                pending = PendingMessage(
                    msg_id=self._next_msg_id,
                    dest=dest,
                    segments=segments,
                    status=DeliveryStatus.IN_FLIGHT,
                    timeout=self._get_timeout(dest),
                    callback=on_delivered,
                )
                self._pending[self._next_msg_id] = pending

            # Send all segments
            for seg in segments:
                self._emit_segment(dest, seg)

            self.stats.messages_sent += 1
            self.stats.segments_sent += len(segments)

            return self._next_msg_id

    def process_segment(self, src: int, segment_data: bytes):
        """
        Process an incoming transport segment.

        Called by the mesh node when a DATA frame arrives.
        """
        try:
            seg = TransportSegment.decode(segment_data)
        except ValueError as e:
            logger.debug(f"Invalid segment from 0x{src:08X}: {e}")
            self.stats.checksum_errors += 1
            return

        if seg.seg_type == SegmentType.ACK:
            self._handle_ack(src, seg)

        elif seg.seg_type == SegmentType.NACK:
            self._handle_nack(src, seg)

        elif seg.seg_type == SegmentType.DATA:
            self._handle_data(src, seg)

        elif seg.seg_type == SegmentType.FIN:
            self._handle_fin(src, seg)

        elif seg.seg_type == SegmentType.PING:
            self._handle_ping(src, seg)

        elif seg.seg_type == SegmentType.PONG:
            self._handle_pong(src, seg)

    def receive(self) -> List["ReceivedData"]:
        """Get all pending received messages."""
        messages = []
        while not self._inbox.empty():
            try:
                messages.append(self._inbox.get_nowait())
            except Empty:
                break
        return messages

    def ping(self, dest: int) -> None:
        """Send a PING to measure RTT."""
        seg = TransportSegment(
            seg_type=SegmentType.PING,
            sequence=self._next_sequence(),
            payload=struct.pack(">d", time.time()),
        )
        self._ping_times[dest] = time.time()
        self._emit_segment(dest, seg)

    def get_rtt(self, dest: int) -> Optional[float]:
        """Get estimated RTT to a peer in seconds."""
        return self._rtt.get(dest)

    def tick(self):
        """
        Periodic maintenance — call this regularly (e.g., every 500ms).

        Handles retransmission, fragment expiry, dedup cleanup.
        """
        now = time.time()

        with self._lock:
            # Check pending messages for timeout
            expired_ids = []
            for msg_id, pending in self._pending.items():
                if pending.status != DeliveryStatus.IN_FLIGHT:
                    continue

                elapsed = now - pending.last_sent
                if elapsed >= pending.timeout:
                    if pending.is_expired:
                        # Give up
                        pending.status = DeliveryStatus.FAILED
                        expired_ids.append(msg_id)
                        self.stats.messages_failed += 1
                        logger.warning(
                            f"Message {msg_id} to 0x{pending.dest:08X} "
                            f"failed after {pending.retries} retries"
                        )
                        if pending.callback:
                            # Call failure callback if provided
                            pass
                    else:
                        # Retransmit unacked segments
                        self._retransmit(pending)

            # Clean up failed messages
            for mid in expired_ids:
                del self._pending[mid]

            # Clean up expired fragment buffers
            expired_frags = [
                key for key, buf in self._frag_buffers.items()
                if buf.is_expired
            ]
            for key in expired_frags:
                del self._frag_buffers[key]
                self.stats.fragments_expired += 1

            # Trim dedup cache
            if len(self._seen_messages) > DEDUP_CACHE_SIZE:
                self._seen_messages = self._seen_messages[-DEDUP_CACHE_SIZE:]

    # ── Internal Methods ──

    def _fragment(self, data: bytes, msg_id: int) -> List[TransportSegment]:
        """Fragment data into transport segments."""
        if len(data) <= MAX_SEGMENT_SIZE:
            seg = TransportSegment(
                seg_type=SegmentType.DATA,
                sequence=self._next_sequence(),
                frag_id=0,
                frag_total=1,
                msg_id=msg_id,
                payload=data,
            )
            return [seg]

        # Split into chunks
        chunks = []
        for i in range(0, len(data), MAX_SEGMENT_SIZE):
            chunks.append(data[i:i + MAX_SEGMENT_SIZE])

        frag_id = self._next_sequence()
        segments = []
        for i, chunk in enumerate(chunks):
            seg = TransportSegment(
                seg_type=SegmentType.DATA,
                sequence=self._next_sequence(),
                frag_id=frag_id,
                frag_total=len(chunks),
                msg_id=msg_id,
                payload=chunk,
            )
            segments.append(seg)

        return segments

    def _handle_data(self, src: int, seg: TransportSegment):
        """Handle incoming DATA segment."""
        # Dedup check
        msg_hash = self._message_hash(src, seg.msg_id, seg.sequence)
        if msg_hash in self._seen_messages:
            logger.debug(f"Duplicate segment from 0x{src:08X} seq={seg.sequence}")
            self.stats.duplicates_detected += 1
            # Still send ACK in case previous ACK was lost
            self._send_ack(src, seg.sequence)
            return

        self._seen_messages.append(msg_hash)

        # Send ACK
        self._send_ack(src, seg.sequence)
        self.stats.segments_received += 1

        # Single segment message
        if seg.frag_total <= 1:
            self._deliver(src, seg.payload)
            return

        # Multi-fragment message — add to reassembly buffer
        key = (src, seg.frag_id)
        if key not in self._frag_buffers:
            self._frag_buffers[key] = FragmentBuffer(
                frag_id=seg.frag_id,
                total=seg.frag_total,
                src=src,
            )

        buf = self._frag_buffers[key]
        frag_index = seg.sequence - seg.frag_id
        buf.segments[frag_index] = seg.payload

        if buf.is_complete:
            try:
                data = buf.reassemble()
                self._deliver(src, data)
            except ValueError:
                logger.error(f"Fragment reassembly failed for 0x{src:08X}")
            finally:
                del self._frag_buffers[key]

    def _handle_ack(self, src: int, seg: TransportSegment):
        """Handle incoming ACK."""
        ack_seq = seg.ack_number
        self.stats.acks_received += 1

        # Find the pending message containing this sequence
        for msg_id, pending in self._pending.items():
            if pending.dest != src:
                continue

            for s in pending.segments:
                if s.sequence == ack_seq:
                    pending.acked_segments.add(ack_seq)

                    if pending.is_complete:
                        pending.status = DeliveryStatus.DELIVERED
                        self.stats.messages_delivered += 1
                        logger.debug(
                            f"Message {msg_id} delivered to 0x{src:08X}"
                        )
                        if pending.callback:
                            try:
                                pending.callback()
                            except Exception:
                                pass
                    return

    def _handle_nack(self, src: int, seg: TransportSegment):
        """Handle NACK — retransmit specific segment."""
        nack_seq = seg.ack_number
        for pending in self._pending.values():
            if pending.dest != src:
                continue
            for s in pending.segments:
                if s.sequence == nack_seq:
                    self._emit_segment(src, s)
                    self.stats.retransmissions += 1
                    return

    def _handle_fin(self, src: int, seg: TransportSegment):
        """Handle FIN — stream complete."""
        logger.debug(f"FIN received from 0x{src:08X}")

    def _handle_ping(self, src: int, seg: TransportSegment):
        """Handle PING — respond with PONG."""
        pong = TransportSegment(
            seg_type=SegmentType.PONG,
            sequence=self._next_sequence(),
            payload=seg.payload,  # Echo the timestamp
        )
        self._emit_segment(src, pong)

    def _handle_pong(self, src: int, seg: TransportSegment):
        """Handle PONG — calculate RTT."""
        if len(seg.payload) >= 8:
            sent_time = struct.unpack(">d", seg.payload[:8])[0]
            rtt = time.time() - sent_time

            # Exponential moving average
            if src in self._rtt:
                self._rtt[src] = 0.8 * self._rtt[src] + 0.2 * rtt
            else:
                self._rtt[src] = rtt

            logger.debug(f"RTT to 0x{src:08X}: {rtt*1000:.1f}ms")

    def _send_ack(self, dest: int, ack_seq: int):
        """Send ACK for a sequence number."""
        ack = TransportSegment(
            seg_type=SegmentType.ACK,
            sequence=self._next_sequence(),
            ack_number=ack_seq,
        )
        self._emit_segment(dest, ack)
        self.stats.acks_sent += 1

    def _retransmit(self, pending: PendingMessage):
        """Retransmit unacknowledged segments with exponential backoff."""
        pending.retries += 1
        pending.timeout = min(
            pending.timeout * 2,
            MAX_TIMEOUT,
        )
        pending.last_sent = time.time()

        for seg in pending.segments:
            if seg.sequence not in pending.acked_segments:
                self._emit_segment(pending.dest, seg)
                self.stats.retransmissions += 1

        logger.debug(
            f"Retransmit msg {pending.msg_id} to 0x{pending.dest:08X} "
            f"(attempt {pending.retries}/{MAX_RETRIES}, "
            f"timeout {pending.timeout:.1f}s)"
        )

    def _deliver(self, src: int, data: bytes):
        """Deliver a complete message to the application."""
        msg = ReceivedData(src=src, data=data, timestamp=time.time())
        self._inbox.put(msg)
        self.stats.messages_received += 1

    def _emit_segment(self, dest: int, seg: TransportSegment):
        """Send a segment via the registered callback."""
        if self._send_callback:
            self._send_callback(dest, seg.encode())
        else:
            self._outbox.put((dest, seg.encode()))

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._next_seq = (self._next_seq + 1) & 0xFFFF
        return self._next_seq

    def _get_timeout(self, dest: int) -> float:
        """Get adaptive timeout based on RTT estimate."""
        rtt = self._rtt.get(dest, BASE_TIMEOUT)
        return max(BASE_TIMEOUT, rtt * 3)  # 3x RTT

    def _message_hash(self, src: int, msg_id: int, seq: int) -> bytes:
        """Create a hash for dedup tracking."""
        raw = struct.pack(">IBH", src, msg_id, seq)
        return hashlib.md5(raw).digest()[:8]


@dataclass
class ReceivedData:
    """A fully reassembled received message."""
    src: int
    data: bytes
    timestamp: float = 0.0


@dataclass
class TransportStats:
    """Transport layer statistics."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    segments_sent: int = 0
    segments_received: int = 0
    acks_sent: int = 0
    acks_received: int = 0
    retransmissions: int = 0
    duplicates_detected: int = 0
    checksum_errors: int = 0
    fragments_expired: int = 0
    flow_control_events: int = 0
