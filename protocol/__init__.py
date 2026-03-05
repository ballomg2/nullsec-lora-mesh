"""
NullSec LoRa Mesh - Frame Protocol

Implements the mesh frame format for encoding/decoding
LoRa mesh protocol frames.
"""

import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional


# Frame sync word
SYNC_WORD = 0x4E53  # "NS" for NullSec

# Protocol version
PROTOCOL_VERSION = 1

# Maximum payload size (LoRa max 255 - header overhead)
MAX_PAYLOAD_SIZE = 222
MAX_FRAME_SIZE = 255


class MessageType(IntEnum):
    """Mesh protocol message types."""
    DATA = 0x01
    ACK = 0x02
    RREQ = 0x03      # Route Request
    RREP = 0x04      # Route Reply
    RERR = 0x05      # Route Error
    HELLO = 0x06     # Neighbor beacon
    KEXCH = 0x07     # Key Exchange
    FRAG = 0x08      # Fragment
    PING = 0x09      # Keepalive
    CTRL = 0x0A      # Control


class FrameFlags(IntEnum):
    """Frame flags bitfield."""
    ENCRYPTED = 0x01
    COMPRESSED = 0x02
    FRAGMENTED = 0x04
    RELIABLE = 0x08   # Request ACK
    BROADCAST = 0x10
    PRIORITY = 0x20


@dataclass
class MeshFrame:
    """
    LoRa Mesh Protocol Frame.

    Header: 18 bytes
    ┌──────┬──────┬──────┬───────┬──────┬──────┬─────┬───────┐
    │ Sync │ Ver  │ Type │ Flags │ Src  │ Dst  │ Seq │ Len   │
    │ 2B   │ 1B   │ 1B   │ 1B    │ 4B   │ 4B   │ 4B  │ 1B    │
    └──────┴──────┴──────┴───────┴──────┴──────┴─────┴───────┘
    """

    msg_type: MessageType
    src_id: int          # 4-byte node ID
    dst_id: int          # 4-byte destination (0xFFFFFFFF = broadcast)
    sequence: int = 0    # 4-byte sequence number
    flags: int = 0       # 1-byte flags bitfield
    payload: bytes = b""
    auth_tag: bytes = b""  # 16-byte AEAD auth tag (if encrypted)

    # Computed
    version: int = PROTOCOL_VERSION

    HEADER_FORMAT = ">HBBB II I B"  # 18 bytes
    HEADER_SIZE = struct.calcsize(">HBBB II I B")  # = 18

    BROADCAST_ADDR = 0xFFFFFFFF

    def encode(self) -> bytes:
        """Encode frame to bytes for transmission."""
        header = struct.pack(
            self.HEADER_FORMAT,
            SYNC_WORD,
            self.version,
            int(self.msg_type),
            self.flags,
            self.src_id,
            self.dst_id,
            self.sequence,
            len(self.payload),
        )

        frame = header + self.payload
        if self.auth_tag:
            frame += self.auth_tag

        if len(frame) > MAX_FRAME_SIZE:
            raise ValueError(
                f"Frame too large: {len(frame)} > {MAX_FRAME_SIZE}"
            )

        return frame

    @classmethod
    def decode(cls, data: bytes) -> "MeshFrame":
        """Decode a frame from received bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Frame too short: {len(data)} < {cls.HEADER_SIZE}")

        (
            sync, version, msg_type, flags,
            src_id, dst_id,
            sequence,
            payload_len,
        ) = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])

        if sync != SYNC_WORD:
            raise ValueError(f"Invalid sync word: 0x{sync:04X}")

        if version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        payload_start = cls.HEADER_SIZE
        payload_end = payload_start + payload_len
        payload = data[payload_start:payload_end]

        # Extract auth tag if encrypted
        auth_tag = b""
        if flags & FrameFlags.ENCRYPTED:
            auth_tag = data[payload_end:payload_end + 16]

        return cls(
            msg_type=MessageType(msg_type),
            src_id=src_id,
            dst_id=dst_id,
            sequence=sequence,
            flags=flags,
            payload=payload,
            auth_tag=auth_tag,
            version=version,
        )

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & FrameFlags.ENCRYPTED)

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & FrameFlags.COMPRESSED)

    @property
    def is_broadcast(self) -> bool:
        return self.dst_id == self.BROADCAST_ADDR or bool(self.flags & FrameFlags.BROADCAST)

    @property
    def is_reliable(self) -> bool:
        return bool(self.flags & FrameFlags.RELIABLE)

    @property
    def total_size(self) -> int:
        return self.HEADER_SIZE + len(self.payload) + len(self.auth_tag)

    def __repr__(self) -> str:
        return (
            f"MeshFrame(type={self.msg_type.name}, "
            f"src=0x{self.src_id:08X}, dst=0x{self.dst_id:08X}, "
            f"seq={self.sequence}, payload={len(self.payload)}B)"
        )


def create_data_frame(
    src: int, dst: int, data: bytes,
    sequence: int = 0, encrypted: bool = True,
    compressed: bool = False, reliable: bool = True,
) -> MeshFrame:
    """Helper to create a DATA frame."""
    flags = 0
    if encrypted:
        flags |= FrameFlags.ENCRYPTED
    if compressed:
        flags |= FrameFlags.COMPRESSED
    if reliable:
        flags |= FrameFlags.RELIABLE
    if dst == MeshFrame.BROADCAST_ADDR:
        flags |= FrameFlags.BROADCAST

    return MeshFrame(
        msg_type=MessageType.DATA,
        src_id=src,
        dst_id=dst,
        sequence=sequence,
        flags=flags,
        payload=data,
    )


def create_ack_frame(src: int, dst: int, ack_seq: int) -> MeshFrame:
    """Create an ACK frame for a given sequence number."""
    return MeshFrame(
        msg_type=MessageType.ACK,
        src_id=src,
        dst_id=dst,
        sequence=ack_seq,
        payload=struct.pack(">I", ack_seq),
    )


def create_hello_frame(src: int, hop_count: int = 0) -> MeshFrame:
    """Create a HELLO beacon frame."""
    return MeshFrame(
        msg_type=MessageType.HELLO,
        src_id=src,
        dst_id=MeshFrame.BROADCAST_ADDR,
        flags=FrameFlags.BROADCAST,
        payload=struct.pack(">B", hop_count),
    )
