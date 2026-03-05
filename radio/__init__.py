"""
NullSec LoRa Mesh - Radio Drivers

Hardware abstraction layer for LoRa radio modules.

Supported hardware:
- SX1276/SX1278 (HopeRF RFM95/96/97/98) — Sub-GHz LoRa v1
- SX1262 — Sub-GHz LoRa v2 (lower power, better performance)

Channel management:
- CSMA/CA with listen-before-talk
- Frequency hopping (PRNG-seeded)
- Duty cycle enforcement (EU/US regulations)
- Adaptive Data Rate (ADR)
"""

from radio.hal import (
    RadioInterface,
    RadioConfig,
    RadioState,
    RadioStats,
    Bandwidth,
    SpreadingFactor,
    CodingRate,
    SX1276Driver,
    SX1262Driver,
)

from radio.channel import (
    ChannelManager,
    ChannelPlan,
    Region,
    CSMA,
    FrequencyHopper,
    AdaptiveDataRate,
)

__all__ = [
    # HAL
    "RadioInterface",
    "RadioConfig",
    "RadioState",
    "RadioStats",
    "Bandwidth",
    "SpreadingFactor",
    "CodingRate",
    "SX1276Driver",
    "SX1262Driver",
    # Channel management
    "ChannelManager",
    "ChannelPlan",
    "Region",
    "CSMA",
    "FrequencyHopper",
    "AdaptiveDataRate",
]
