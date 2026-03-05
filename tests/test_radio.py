"""
Tests for radio HAL drivers and configuration.
"""

import pytest
import struct
import time

# Test RadioConfig
from radio.hal import (
    RadioConfig, RadioState, RadioStats,
    Bandwidth, SpreadingFactor, CodingRate,
    SX1276Driver, SX1262Driver,
    RadioInterface,
)


class TestRadioConfig:
    """Test radio configuration calculations."""

    def test_default_config(self):
        config = RadioConfig()
        assert config.frequency == 915_000_000
        assert config.bandwidth == Bandwidth.BW_125000
        assert config.spreading_factor == SpreadingFactor.SF7
        assert config.coding_rate == CodingRate.CR_4_5
        assert config.tx_power == 17
        assert config.sync_word == 0x4E53  # NullSec sync word
        assert config.crc_enabled is True

    def test_symbol_rate_sf7_bw125(self):
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF7,
            bandwidth=Bandwidth.BW_125000,
        )
        expected = 125000 / (2 ** 7)  # ~976.56 symbols/sec
        assert abs(config.symbol_rate - expected) < 0.01

    def test_symbol_rate_sf12_bw125(self):
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF12,
            bandwidth=Bandwidth.BW_125000,
        )
        expected = 125000 / (2 ** 12)  # ~30.52 symbols/sec
        assert abs(config.symbol_rate - expected) < 0.01

    def test_bit_rate_sf7(self):
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF7,
            bandwidth=Bandwidth.BW_125000,
            coding_rate=CodingRate.CR_4_5,
        )
        # SF7 * (BW / 2^SF) * (4/CR)
        expected = 7 * (125000 / 128) * (4 / 5)
        assert abs(config.bit_rate - expected) < 0.01

    def test_low_data_rate_optimize_auto(self):
        """SF11 and SF12 with BW125 should auto-enable LDRO."""
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF12,
            bandwidth=Bandwidth.BW_125000,
        )
        assert config.low_data_rate_optimize is True

    def test_low_data_rate_optimize_not_needed(self):
        """SF7 with BW125 should not need LDRO."""
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF7,
            bandwidth=Bandwidth.BW_125000,
        )
        assert config.low_data_rate_optimize is False

    def test_time_on_air_small_packet(self):
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF7,
            bandwidth=Bandwidth.BW_125000,
        )
        toa = config.time_on_air(10)  # 10 byte payload
        assert 0 < toa < 1.0  # Should be well under 1 second

    def test_time_on_air_max_packet(self):
        config = RadioConfig(
            spreading_factor=SpreadingFactor.SF12,
            bandwidth=Bandwidth.BW_125000,
        )
        toa = config.time_on_air(222)  # Max mesh payload
        assert toa > 1.0  # SF12 max payload should take > 1 second

    def test_time_on_air_increases_with_sf(self):
        toa_sf7 = RadioConfig(spreading_factor=SpreadingFactor.SF7).time_on_air(50)
        toa_sf12 = RadioConfig(spreading_factor=SpreadingFactor.SF12).time_on_air(50)
        assert toa_sf12 > toa_sf7

    def test_max_payload_implicit(self):
        config = RadioConfig(implicit_header=True)
        assert config.max_payload == 255

    def test_max_payload_explicit(self):
        config = RadioConfig(implicit_header=False)
        assert config.max_payload == 222


class TestRadioStats:
    """Test radio statistics."""

    def test_default_stats(self):
        stats = RadioStats()
        assert stats.tx_packets == 0
        assert stats.rx_packets == 0
        assert stats.tx_bytes == 0
        assert stats.rx_bytes == 0
        assert stats.crc_errors == 0

    def test_stats_update(self):
        stats = RadioStats()
        stats.tx_packets += 1
        stats.tx_bytes += 100
        assert stats.tx_packets == 1
        assert stats.tx_bytes == 100


class TestSX1276Driver:
    """Test SX1276 driver with stub hardware."""

    def test_initialization(self):
        driver = SX1276Driver()
        assert driver.state == RadioState.IDLE

    def test_initialize_with_stub(self):
        driver = SX1276Driver()
        result = driver.initialize()
        assert result is True
        assert driver.state == RadioState.STANDBY

    def test_configure(self):
        driver = SX1276Driver()
        driver.initialize()
        config = RadioConfig(
            frequency=868_000_000,
            spreading_factor=SpreadingFactor.SF9,
            bandwidth=Bandwidth.BW_250000,
        )
        driver.configure(config)
        assert driver.config.frequency == 868_000_000
        assert driver.config.spreading_factor == SpreadingFactor.SF9

    def test_transmit(self):
        driver = SX1276Driver()
        driver.initialize()
        result = driver.transmit(b"Hello LoRa!")
        assert result is True
        assert driver.state == RadioState.TX
        assert driver.stats.tx_packets == 1
        assert driver.stats.tx_bytes == 11

    def test_transmit_too_large(self):
        driver = SX1276Driver()
        driver.initialize()
        result = driver.transmit(b"x" * 256)
        assert result is False

    def test_sleep(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.sleep()
        assert driver.state == RadioState.SLEEP

    def test_standby(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.standby()
        assert driver.state == RadioState.STANDBY

    def test_set_frequency(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.set_frequency(868_100_000)
        assert driver.config.frequency == 868_100_000

    def test_set_tx_power(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.set_tx_power(20)
        assert driver.config.tx_power == 20

    def test_set_tx_power_clamped(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.set_tx_power(25)  # Over max
        assert driver.config.tx_power == 20

    def test_callbacks(self):
        driver = SX1276Driver()
        received = []
        driver.on_receive(lambda data, rssi, snr: received.append(data))
        assert driver._rx_callback is not None

    def test_reset_stats(self):
        driver = SX1276Driver()
        driver.initialize()
        driver.transmit(b"test")
        assert driver.stats.tx_packets == 1
        driver.reset_stats()
        assert driver.stats.tx_packets == 0


class TestSX1262Driver:
    """Test SX1262 driver with stub hardware."""

    def test_initialization(self):
        driver = SX1262Driver()
        assert driver.state == RadioState.IDLE

    def test_initialize_with_stub(self):
        driver = SX1262Driver()
        result = driver.initialize()
        assert result is True
        assert driver.state == RadioState.STANDBY

    def test_transmit(self):
        driver = SX1262Driver()
        driver.initialize()
        result = driver.transmit(b"Hello SX1262!")
        assert result is True
        assert driver.state == RadioState.TX

    def test_sleep(self):
        driver = SX1262Driver()
        driver.initialize()
        driver.sleep()
        assert driver.state == RadioState.SLEEP

    def test_set_frequency(self):
        driver = SX1262Driver()
        driver.initialize()
        driver.set_frequency(915_000_000)
        assert driver.config.frequency == 915_000_000

    def test_set_tx_power_range(self):
        driver = SX1262Driver()
        driver.initialize()
        driver.set_tx_power(22)  # SX1262 supports up to +22
        assert driver.config.tx_power == 22

        driver.set_tx_power(-9)  # Minimum
        assert driver.config.tx_power == -9


class TestRadioInterface:
    """Test the RadioInterface ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RadioInterface()

    def test_subclass_must_implement(self):
        class BadDriver(RadioInterface):
            pass

        with pytest.raises(TypeError):
            BadDriver()
