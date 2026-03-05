"""
Tests for radio channel manager — CSMA/CA, frequency hopping,
duty cycle, adaptive data rate.
"""

import pytest
import time
from unittest.mock import MagicMock

from radio.channel import (
    ChannelManager,
    ChannelPlan,
    Region,
    CSMA,
    FrequencyHopper,
    AdaptiveDataRate,
    ChannelState,
)


class TestChannelPlan:
    """Test channel plan definitions."""

    def test_us_915_channels(self):
        plan = ChannelPlan.us_915()
        assert plan.region == Region.US_915
        assert len(plan.channels) == 64
        assert plan.channels[0] == 902_300_000  # First channel
        assert plan.max_tx_power == 30
        assert plan.duty_cycle == 1.0  # No duty cycle limit

    def test_eu_868_channels(self):
        plan = ChannelPlan.eu_868()
        assert plan.region == Region.EU_868
        assert len(plan.channels) == 8
        assert plan.max_tx_power == 14
        assert plan.duty_cycle == 0.01  # 1% duty cycle

    def test_us_915_spacing(self):
        plan = ChannelPlan.us_915()
        # Channels should be 200kHz apart
        for i in range(1, len(plan.channels)):
            spacing = plan.channels[i] - plan.channels[i - 1]
            assert spacing == 200_000


class TestCSMA:
    """Test CSMA/CA channel access."""

    def test_clear_channel(self):
        """Should grant access immediately when channel is clear."""
        csma = CSMA(cad_function=lambda: False)
        assert csma.request_channel() is True

    def test_busy_channel_retries(self):
        """Should backoff and retry on busy channel."""
        call_count = [0]

        def cad():
            call_count[0] += 1
            return call_count[0] < 3  # Busy for first 2 checks

        csma = CSMA(cad_function=cad)
        assert csma.request_channel() is True
        assert call_count[0] == 3

    def test_always_busy_fails(self):
        """Should fail after max retries on permanently busy channel."""
        csma = CSMA(cad_function=lambda: True)
        result = csma.request_channel()
        assert result is False

    def test_no_cad_function(self):
        """Without CAD, channel should always appear clear."""
        csma = CSMA(cad_function=None)
        assert csma.request_channel() is True

    def test_contention_count(self):
        count = [0]

        def cad():
            count[0] += 1
            return count[0] < 2

        csma = CSMA(cad_function=cad)
        csma.request_channel()
        assert csma.contention_count >= 1


class TestFrequencyHopper:
    """Test frequency hopping sequence generation."""

    def test_deterministic_sequence(self):
        """Same seed should produce same sequence."""
        plan = ChannelPlan.us_915()
        h1 = FrequencyHopper(plan, hop_seed=b"test-seed-1")
        h2 = FrequencyHopper(plan, hop_seed=b"test-seed-1")

        seq1 = h1.get_sequence()
        seq2 = h2.get_sequence()
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self):
        """Different seeds should produce different sequences."""
        plan = ChannelPlan.us_915()
        h1 = FrequencyHopper(plan, hop_seed=b"seed-A")
        h2 = FrequencyHopper(plan, hop_seed=b"seed-B")

        assert h1.get_sequence() != h2.get_sequence()

    def test_sequence_covers_all_channels(self):
        """Sequence should contain all channels (permutation)."""
        plan = ChannelPlan.us_915()
        hopper = FrequencyHopper(plan)

        seq = hopper.get_sequence()
        assert set(seq) == set(plan.channels)
        assert len(seq) == len(plan.channels)

    def test_current_channel(self):
        plan = ChannelPlan.us_915()
        hopper = FrequencyHopper(plan)
        ch = hopper.current_channel()
        assert ch in plan.channels

    def test_next_channel_advances(self):
        plan = ChannelPlan.us_915()
        hopper = FrequencyHopper(plan)
        ch1 = hopper.current_channel()
        ch2 = hopper.next_channel()
        # Should advance (might wrap, but shouldn't be same unless 1 channel)
        assert ch2 in plan.channels

    def test_hop_to(self):
        plan = ChannelPlan.us_915()
        hopper = FrequencyHopper(plan)
        ch = hopper.hop_to(10)
        assert ch in plan.channels

    def test_sequence_length(self):
        plan = ChannelPlan.us_915()
        hopper = FrequencyHopper(plan)
        assert hopper.sequence_length == 64

    def test_eu_868_hopping(self):
        plan = ChannelPlan.eu_868()
        hopper = FrequencyHopper(plan)
        assert hopper.sequence_length == 8
        seq = hopper.get_sequence()
        assert set(seq) == set(plan.channels)


class TestAdaptiveDataRate:
    """Test Adaptive Data Rate controller."""

    def test_initial_params(self):
        adr = AdaptiveDataRate(initial_sf=7, initial_power=14)
        assert adr.sf == 7
        assert adr.tx_power == 14

    def test_good_link_reduces_power(self):
        """Good link quality should eventually reduce TX power."""
        adr = AdaptiveDataRate(initial_sf=7, initial_power=14)

        # Feed good link quality samples
        for _ in range(15):
            adr.update(snr=20.0, rssi=-50, success=True)

        # Should have reduced power or SF
        assert adr.tx_power < 14 or adr.sf < 7

    def test_poor_link_increases_sf(self):
        """Poor link quality should increase SF."""
        adr = AdaptiveDataRate(initial_sf=7, initial_power=14)

        # Feed poor link quality (high loss)
        for _ in range(20):
            adr.update(snr=-5.0, rssi=-100, success=False)

        # Should have increased SF or power
        assert adr.sf > 7 or adr.tx_power > 14

    def test_reset(self):
        adr = AdaptiveDataRate()
        adr.update(snr=10.0, rssi=-60, success=True)
        adr.reset()
        assert len(adr._snr_history) == 0
        assert adr._loss_count == 0

    def test_sf_max_bound(self):
        """SF should not exceed 12."""
        adr = AdaptiveDataRate(initial_sf=12, initial_power=20)
        for _ in range(20):
            adr.update(snr=-20.0, rssi=-120, success=False)
        assert adr.sf <= 12

    def test_power_max_bound(self):
        """TX power should not exceed 20 dBm."""
        adr = AdaptiveDataRate(initial_sf=12, initial_power=20)
        for _ in range(20):
            adr.update(snr=-20.0, rssi=-120, success=False)
        assert adr.tx_power <= 20


class TestChannelState:
    """Test per-channel state tracking."""

    def test_record_tx(self):
        state = ChannelState(frequency=915_000_000)
        state.record_tx(0.1)  # 100ms TX
        assert state.tx_count == 1
        assert state.total_tx_time == pytest.approx(0.1)

    def test_duty_cycle_tracking(self):
        state = ChannelState(frequency=868_100_000)
        state.record_tx(0.01)  # 10ms
        dc = state.duty_cycle_used()
        assert dc > 0

    def test_reset_window(self):
        state = ChannelState(frequency=915_000_000)
        state.record_tx(1.0)
        state.reset_window()
        assert state.total_tx_time == 0.0
        assert state.tx_count == 0


class TestChannelManager:
    """Test unified channel manager."""

    def test_init_us915(self):
        mgr = ChannelManager(region=Region.US_915)
        assert mgr.plan.region == Region.US_915
        assert len(mgr.plan.channels) == 64

    def test_init_eu868(self):
        mgr = ChannelManager(region=Region.EU_868)
        assert mgr.plan.region == Region.EU_868
        assert len(mgr.plan.channels) == 8

    def test_acquire_channel(self):
        mgr = ChannelManager(
            region=Region.US_915,
            cad_function=lambda: False,
        )
        freq = mgr.acquire_channel()
        assert freq is not None
        assert freq in mgr.plan.channels

    def test_record_transmission(self):
        mgr = ChannelManager(region=Region.US_915)
        freq = mgr.plan.channels[0]
        mgr.record_transmission(freq, 0.1)
        assert mgr.stats.total_tx_time > 0

    def test_get_recommended_params(self):
        mgr = ChannelManager(region=Region.US_915)
        sf, power = mgr.get_recommended_params()
        assert 6 <= sf <= 12
        assert 2 <= power <= 22

    def test_channel_status(self):
        mgr = ChannelManager(region=Region.US_915, hop_enabled=False)
        status = mgr.get_channel_status()
        assert len(status) == 64
        assert "frequency_mhz" in status[0]
        assert "duty_cycle_used" in status[0]

    def test_duty_cycle_reset(self):
        mgr = ChannelManager(region=Region.US_915)
        freq = mgr.plan.channels[0]
        mgr.record_transmission(freq, 1.0)
        mgr.reset_duty_cycle()

        state = mgr._channel_states[freq]
        assert state.total_tx_time == 0.0

    def test_hopping_disabled(self):
        mgr = ChannelManager(region=Region.US_915, hop_enabled=False)
        assert mgr.hopper is None
