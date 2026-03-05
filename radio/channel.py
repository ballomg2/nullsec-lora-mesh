"""
NullSec LoRa Mesh - Channel Manager

Manages RF channel access with:
- CSMA/CA (Carrier Sense Multiple Access / Collision Avoidance)
- Frequency hopping for interference avoidance and regulatory compliance
- Duty cycle enforcement (EU 868MHz: 1%, US 915MHz: flexible)
- Adaptive data rate (ADR) selection
- Listen-Before-Talk (LBT) with backoff
"""

import time
import random
import hashlib
import struct
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Region(IntEnum):
    """Regulatory region for channel plan."""
    US_915 = 0       # 902-928 MHz ISM (FCC Part 15)
    EU_868 = 1       # 863-870 MHz (ETSI)
    AU_915 = 2       # 915-928 MHz (ACMA)
    AS_923 = 3       # 920-923 MHz (Asia)
    KR_920 = 4       # 920-923 MHz (Korea)
    IN_865 = 5       # 865-867 MHz (India)
    CUSTOM = 0xFF    # User-defined


@dataclass
class ChannelPlan:
    """Defines available channels for a regulatory region."""
    region: Region
    channels: List[int]           # Frequencies in Hz
    max_tx_power: int             # dBm
    duty_cycle: float             # Max duty cycle (0.0-1.0)
    dwell_time_ms: int            # Max dwell time per channel (ms)
    min_channel_spacing: int      # Hz between channels

    @staticmethod
    def us_915() -> "ChannelPlan":
        """US ISM 915 MHz plan (64 uplink + 8 downlink channels)."""
        channels = [
            902_300_000 + i * 200_000 for i in range(64)  # 200kHz spacing
        ]
        return ChannelPlan(
            region=Region.US_915,
            channels=channels,
            max_tx_power=30,       # +30 dBm EIRP
            duty_cycle=1.0,        # No duty cycle limit
            dwell_time_ms=400,     # 400ms max dwell
            min_channel_spacing=200_000,
        )

    @staticmethod
    def eu_868() -> "ChannelPlan":
        """EU 868 MHz plan."""
        channels = [
            868_100_000,  # g1: 868.0-868.6 (1% DC)
            868_300_000,
            868_500_000,
            867_100_000,  # g: 867.0-868.0 (1% DC)
            867_300_000,
            867_500_000,
            867_700_000,
            867_900_000,
        ]
        return ChannelPlan(
            region=Region.EU_868,
            channels=channels,
            max_tx_power=14,       # +14 dBm ERP
            duty_cycle=0.01,       # 1% duty cycle
            dwell_time_ms=0,       # No dwell limit
            min_channel_spacing=200_000,
        )


@dataclass
class ChannelState:
    """Tracks per-channel usage for duty cycle enforcement."""
    frequency: int
    total_tx_time: float = 0.0    # Cumulative TX time in seconds
    last_tx_start: float = 0.0
    last_tx_end: float = 0.0
    window_start: float = field(default_factory=time.time)
    tx_count: int = 0
    avg_rssi: float = -120.0      # Background noise level

    def record_tx(self, duration: float):
        """Record a transmission on this channel."""
        now = time.time()
        self.last_tx_start = now
        self.last_tx_end = now + duration
        self.total_tx_time += duration
        self.tx_count += 1

    def duty_cycle_used(self) -> float:
        """Calculate current duty cycle usage in the observation window."""
        window = time.time() - self.window_start
        if window <= 0:
            return 0.0
        return self.total_tx_time / window

    def reset_window(self):
        """Reset the duty cycle observation window."""
        self.total_tx_time = 0.0
        self.window_start = time.time()
        self.tx_count = 0


class CSMA:
    """
    CSMA/CA implementation for LoRa channel access.

    Uses listen-before-talk with random backoff to avoid
    collisions on shared channels.
    """

    # CSMA parameters
    MIN_BACKOFF_MS = 10
    MAX_BACKOFF_MS = 500
    BACKOFF_EXPONENT_MAX = 5
    CCA_THRESHOLD = -80   # dBm — channel considered busy above this

    def __init__(self, cad_function: Optional[Callable[[], bool]] = None):
        """
        Args:
            cad_function: Function that performs Channel Activity Detection.
                          Returns True if channel is busy.
        """
        self._cad = cad_function
        self._backoff_exponent = 0
        self._contention_count = 0

    def request_channel(self) -> bool:
        """
        Request access to the channel using CSMA/CA.

        Returns True if the channel is clear to transmit.
        Blocks with random backoff if channel is busy.
        """
        for attempt in range(self.BACKOFF_EXPONENT_MAX + 1):
            # Listen Before Talk
            if not self._channel_busy():
                self._backoff_exponent = 0
                return True

            # Channel busy — backoff
            self._backoff_exponent = min(
                self._backoff_exponent + 1,
                self.BACKOFF_EXPONENT_MAX,
            )

            backoff_ms = random.randint(
                self.MIN_BACKOFF_MS,
                min(
                    self.MIN_BACKOFF_MS * (2 ** self._backoff_exponent),
                    self.MAX_BACKOFF_MS,
                ),
            )

            logger.debug(
                f"CSMA backoff: {backoff_ms}ms (attempt {attempt + 1})"
            )
            time.sleep(backoff_ms / 1000)
            self._contention_count += 1

        logger.warning("CSMA: channel access failed after max retries")
        return False

    def _channel_busy(self) -> bool:
        """Check if channel is currently busy."""
        if self._cad:
            return self._cad()
        return False

    @property
    def contention_count(self) -> int:
        """Number of times backoff was triggered."""
        return self._contention_count


class FrequencyHopper:
    """
    Pseudo-random frequency hopping for interference avoidance.

    Uses a shared seed to synchronize hopping sequences
    across mesh nodes without explicit coordination.
    """

    def __init__(self, channel_plan: ChannelPlan,
                 hop_seed: bytes = b"nullsec-mesh-hop"):
        self.plan = channel_plan
        self.seed = hop_seed
        self._hop_index = 0
        self._sequence = self._generate_sequence()

    def current_channel(self) -> int:
        """Get the current hopping channel frequency in Hz."""
        idx = self._hop_index % len(self._sequence)
        return self._sequence[idx]

    def next_channel(self) -> int:
        """Advance to the next channel in the hopping sequence."""
        self._hop_index += 1
        return self.current_channel()

    def hop_to(self, index: int) -> int:
        """Jump to a specific position in the hopping sequence."""
        self._hop_index = index
        return self.current_channel()

    def sync_to_time(self, slot_duration_ms: int = 400) -> int:
        """
        Synchronize hopping to time slots.

        All nodes with the same seed and slot duration will be
        on the same channel at the same time.
        """
        now_ms = int(time.time() * 1000)
        slot = now_ms // slot_duration_ms
        self._hop_index = slot
        return self.current_channel()

    def _generate_sequence(self) -> List[int]:
        """
        Generate a pseudo-random permutation of channels.

        Uses the shared seed to create a deterministic but
        well-distributed hopping pattern.
        """
        channels = list(self.plan.channels)
        n = len(channels)

        # Fisher-Yates shuffle with seeded PRNG
        rng = random.Random(self.seed)
        for i in range(n - 1, 0, -1):
            j = rng.randint(0, i)
            channels[i], channels[j] = channels[j], channels[i]

        return channels

    @property
    def sequence_length(self) -> int:
        """Length of the hopping sequence."""
        return len(self._sequence)

    def get_sequence(self) -> List[int]:
        """Get the full hopping sequence (for debugging)."""
        return list(self._sequence)


class AdaptiveDataRate:
    """
    Adaptive Data Rate (ADR) controller.

    Automatically adjusts spreading factor and TX power
    based on link quality metrics (RSSI, SNR, packet loss).

    Goal: Maximize throughput while maintaining reliability.
    """

    # ADR parameters
    SNR_MARGIN = 10.0        # dB margin above demodulation floor
    ADR_ACK_LIMIT = 64       # Frames before forcing ADR
    ADR_ACK_DELAY = 32       # Frames before requesting ADR

    # SNR demodulation floor per SF (approximate)
    SNR_FLOOR = {
        7: -7.5,
        8: -10.0,
        9: -12.5,
        10: -15.0,
        11: -17.5,
        12: -20.0,
    }

    def __init__(self, initial_sf: int = 7, initial_power: int = 14):
        self.sf = initial_sf
        self.tx_power = initial_power
        self._snr_history: List[float] = []
        self._loss_count = 0
        self._frame_count = 0

    def update(self, snr: float, rssi: int, success: bool) -> Tuple[int, int]:
        """
        Update ADR with new link quality sample.

        Args:
            snr: Signal-to-Noise Ratio (dB)
            rssi: Received Signal Strength (dBm)
            success: Whether the packet was successfully received

        Returns:
            (new_sf, new_tx_power) tuple
        """
        self._frame_count += 1
        self._snr_history.append(snr)

        if not success:
            self._loss_count += 1

        # Keep history manageable
        if len(self._snr_history) > 20:
            self._snr_history = self._snr_history[-20:]

        # Calculate average SNR
        avg_snr = sum(self._snr_history) / len(self._snr_history)

        # Calculate packet loss rate
        loss_rate = self._loss_count / max(self._frame_count, 1)

        # Get SNR margin above demodulation floor
        snr_margin = avg_snr - self.SNR_FLOOR.get(self.sf, -7.5)

        old_sf = self.sf
        old_power = self.tx_power

        if loss_rate > 0.1:
            # High loss — increase robustness
            if self.sf < 12:
                self.sf += 1
            elif self.tx_power < 20:
                self.tx_power = min(self.tx_power + 2, 20)
        elif snr_margin > self.SNR_MARGIN and len(self._snr_history) >= 10:
            # Good link — try to optimize
            if self.tx_power > 2:
                self.tx_power = max(self.tx_power - 2, 2)
            elif self.sf > 7:
                self.sf -= 1
                self.tx_power = 14  # Reset power when changing SF

        if self.sf != old_sf or self.tx_power != old_power:
            logger.info(
                f"ADR: SF{old_sf}→SF{self.sf}, "
                f"{old_power}→{self.tx_power}dBm "
                f"(SNR margin: {snr_margin:.1f}dB, loss: {loss_rate:.1%})"
            )
            # Reset counters after adjustment
            self._loss_count = 0
            self._frame_count = 0

        return self.sf, self.tx_power

    def reset(self):
        """Reset ADR state."""
        self._snr_history.clear()
        self._loss_count = 0
        self._frame_count = 0


class ChannelManager:
    """
    Unified channel management combining CSMA/CA, frequency hopping,
    duty cycle enforcement, and adaptive data rate.

    This is the main interface used by the MeshNode for all
    RF channel operations.
    """

    DUTY_CYCLE_WINDOW = 3600  # 1 hour observation window

    def __init__(
        self,
        region: Region = Region.US_915,
        hop_enabled: bool = True,
        hop_seed: bytes = b"nullsec-mesh-hop",
        cad_function: Optional[Callable[[], bool]] = None,
    ):
        # Channel plan
        if region == Region.US_915:
            self.plan = ChannelPlan.us_915()
        elif region == Region.EU_868:
            self.plan = ChannelPlan.eu_868()
        else:
            self.plan = ChannelPlan.us_915()

        # Components
        self.csma = CSMA(cad_function)
        self.hopper = FrequencyHopper(self.plan, hop_seed) if hop_enabled else None
        self.adr = AdaptiveDataRate()

        # Per-channel state
        self._channel_states: Dict[int, ChannelState] = {
            freq: ChannelState(frequency=freq)
            for freq in self.plan.channels
        }

        # Current channel
        self._current_freq = self.plan.channels[0]

        # Lock
        self._lock = threading.Lock()

        # Stats
        self.stats = ChannelStats()

    def acquire_channel(self) -> Optional[int]:
        """
        Acquire a channel for transmission.

        Performs:
        1. Frequency hop (if enabled)
        2. Duty cycle check
        3. CSMA/CA channel access

        Returns:
            Channel frequency (Hz) if acquired, None if denied.
        """
        with self._lock:
            # Step 1: Select channel
            if self.hopper:
                freq = self.hopper.next_channel()
            else:
                freq = self._current_freq

            # Step 2: Duty cycle check
            state = self._channel_states.get(freq)
            if state:
                dc_used = state.duty_cycle_used()
                if dc_used >= self.plan.duty_cycle:
                    # This channel is exhausted — try next
                    logger.debug(
                        f"Channel {freq/1e6:.1f}MHz duty cycle "
                        f"exhausted ({dc_used:.3%})"
                    )
                    freq = self._find_available_channel()
                    if freq is None:
                        self.stats.duty_cycle_denials += 1
                        return None

            self._current_freq = freq

        # Step 3: CSMA/CA (outside lock to allow blocking)
        if self.csma.request_channel():
            self.stats.channel_acquisitions += 1
            return freq
        else:
            self.stats.csma_failures += 1
            return None

    def record_transmission(self, freq: int, duration: float):
        """Record a completed transmission for duty cycle tracking."""
        with self._lock:
            state = self._channel_states.get(freq)
            if state:
                state.record_tx(duration)
                self.stats.total_tx_time += duration

    def update_link_quality(self, snr: float, rssi: int, success: bool):
        """Update ADR with new link quality measurement."""
        self.adr.update(snr, rssi, success)

    def get_recommended_params(self) -> Tuple[int, int]:
        """Get ADR-recommended SF and TX power."""
        return self.adr.sf, self.adr.tx_power

    def get_current_frequency(self) -> int:
        """Get the current operating frequency in Hz."""
        return self._current_freq

    def reset_duty_cycle(self):
        """Reset all channel duty cycle windows."""
        with self._lock:
            for state in self._channel_states.values():
                state.reset_window()

    def get_channel_status(self) -> List[Dict]:
        """Get status of all channels."""
        status = []
        for freq, state in self._channel_states.items():
            status.append({
                "frequency_mhz": freq / 1e6,
                "duty_cycle_used": state.duty_cycle_used(),
                "tx_count": state.tx_count,
                "avg_rssi": state.avg_rssi,
            })
        return status

    def _find_available_channel(self) -> Optional[int]:
        """Find a channel with remaining duty cycle budget."""
        for freq, state in self._channel_states.items():
            if state.duty_cycle_used() < self.plan.duty_cycle:
                return freq

        # All channels exhausted — try resetting if window expired
        now = time.time()
        for freq, state in self._channel_states.items():
            if (now - state.window_start) >= self.DUTY_CYCLE_WINDOW:
                state.reset_window()
                return freq

        return None


@dataclass
class ChannelStats:
    """Channel manager statistics."""
    channel_acquisitions: int = 0
    csma_failures: int = 0
    duty_cycle_denials: int = 0
    total_tx_time: float = 0.0
    frequency_hops: int = 0
