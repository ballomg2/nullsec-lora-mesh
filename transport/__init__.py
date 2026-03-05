"""
NullSec LoRa Mesh - Transport Layer

Reliable, ordered delivery over the LoRa mesh network.

Features:
- Stop-and-wait ARQ with exponential backoff
- Message fragmentation and reassembly
- CRC-16 integrity checking
- Duplicate detection and suppression
- RTT estimation and adaptive timeouts
- Flow control with backpressure
- Delivery confirmation callbacks
"""

from transport.reliable import (
    TransportLayer,
    TransportSegment,
    TransportStats,
    ReceivedData,
    SegmentType,
    DeliveryStatus,
    PendingMessage,
    FragmentBuffer,
    MAX_SEGMENT_SIZE,
    MAX_RETRIES,
    WINDOW_SIZE,
)

__all__ = [
    "TransportLayer",
    "TransportSegment",
    "TransportStats",
    "ReceivedData",
    "SegmentType",
    "DeliveryStatus",
    "PendingMessage",
    "FragmentBuffer",
    "MAX_SEGMENT_SIZE",
    "MAX_RETRIES",
    "WINDOW_SIZE",
]
