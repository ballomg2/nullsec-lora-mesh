"""
NullSec LoRa Mesh - Radio Hardware Abstraction Layer

Provides a unified interface for different LoRa radio modules:
- SX1276/SX1278 (LoRa v1, HopeRF RFM95/96/97/98)
- SX1262 (LoRa v2, Semtech modern chipset)
- SX1280 (2.4GHz LoRa, long range BLE alternative)

All drivers implement the RadioInterface ABC for seamless swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable, List
import struct
import time
import logging

logger = logging.getLogger(__name__)


class RadioState(IntEnum):
    """Radio operating states."""
    IDLE = 0
    TX = 1
    RX = 2
    CAD = 3  # Channel Activity Detection
    SLEEP = 4
    STANDBY = 5


class Bandwidth(IntEnum):
    """LoRa bandwidth options in Hz."""
    BW_7800 = 7800
    BW_10400 = 10400
    BW_15600 = 15600
    BW_20800 = 20800
    BW_31250 = 31250
    BW_41700 = 41700
    BW_62500 = 62500
    BW_125000 = 125000
    BW_250000 = 250000
    BW_500000 = 500000


class SpreadingFactor(IntEnum):
    """LoRa spreading factors (SF6-SF12)."""
    SF6 = 6
    SF7 = 7
    SF8 = 8
    SF9 = 9
    SF10 = 10
    SF11 = 11
    SF12 = 12


class CodingRate(IntEnum):
    """LoRa coding rates."""
    CR_4_5 = 5
    CR_4_6 = 6
    CR_4_7 = 7
    CR_4_8 = 8


@dataclass
class RadioConfig:
    """Radio configuration parameters."""
    frequency: int = 915_000_000       # Hz (915 MHz default, ISM band)
    bandwidth: Bandwidth = Bandwidth.BW_125000
    spreading_factor: SpreadingFactor = SpreadingFactor.SF7
    coding_rate: CodingRate = CodingRate.CR_4_5
    tx_power: int = 17                 # dBm (max 20 for most modules)
    sync_word: int = 0x4E53            # 'NS' - NullSec sync word
    preamble_length: int = 8
    implicit_header: bool = False
    crc_enabled: bool = True
    iq_inverted: bool = False
    lna_gain: int = 0                  # 0 = AGC, 1-6 = manual gain
    ocp_current: int = 100             # Over-current protection (mA)
    low_data_rate_optimize: bool = False  # Auto-set for SF11/SF12

    def __post_init__(self):
        """Auto-configure low data rate optimization."""
        if self.spreading_factor >= SpreadingFactor.SF11:
            symbol_duration = (2 ** self.spreading_factor) / self.bandwidth
            if symbol_duration > 0.016:  # > 16ms
                self.low_data_rate_optimize = True

    @property
    def symbol_rate(self) -> float:
        """Calculate symbol rate in symbols/sec."""
        return self.bandwidth / (2 ** self.spreading_factor)

    @property
    def bit_rate(self) -> float:
        """Calculate effective bit rate in bits/sec."""
        return self.spreading_factor * (
            self.bandwidth / (2 ** self.spreading_factor)
        ) * (4 / self.coding_rate)

    @property
    def max_payload(self) -> int:
        """Maximum payload size for current config."""
        if self.implicit_header:
            return 255
        return 222  # Our mesh frame max payload

    def time_on_air(self, payload_bytes: int) -> float:
        """Calculate time on air in seconds for a given payload size."""
        sf = self.spreading_factor
        bw = self.bandwidth
        cr = self.coding_rate

        t_sym = (2 ** sf) / bw
        t_preamble = (self.preamble_length + 4.25) * t_sym

        # Payload symbol count
        h = 0 if not self.implicit_header else 1
        de = 1 if self.low_data_rate_optimize else 0
        crc_present = 1 if self.crc_enabled else 0

        numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc_present - 20 * h
        denominator = 4 * (sf - 2 * de)

        n_payload = 8 + max(
            ((numerator + denominator - 1) // denominator) * cr, 0
        )

        t_payload = n_payload * t_sym
        return t_preamble + t_payload


@dataclass
class RadioStats:
    """Radio statistics."""
    tx_packets: int = 0
    rx_packets: int = 0
    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_errors: int = 0
    rx_errors: int = 0
    crc_errors: int = 0
    last_rssi: int = 0       # dBm
    last_snr: float = 0.0    # dB
    channel_busy: int = 0    # CAD detections


class RadioInterface(ABC):
    """
    Abstract base class for LoRa radio drivers.
    
    All radio implementations must inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, config: Optional[RadioConfig] = None):
        self.config = config or RadioConfig()
        self.state = RadioState.IDLE
        self.stats = RadioStats()
        self._rx_callback: Optional[Callable] = None
        self._tx_done_callback: Optional[Callable] = None
        self._cad_callback: Optional[Callable] = None

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the radio hardware. Returns True on success."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Hardware reset the radio module."""
        ...

    @abstractmethod
    def configure(self, config: RadioConfig) -> None:
        """Apply radio configuration."""
        ...

    @abstractmethod
    def transmit(self, data: bytes) -> bool:
        """
        Transmit data. Returns True if transmission started successfully.
        Non-blocking — use on_tx_done callback for completion.
        """
        ...

    @abstractmethod
    def receive(self, timeout_ms: int = 0) -> Optional[bytes]:
        """
        Start receiving. timeout_ms=0 for continuous RX.
        Returns received data if blocking, None if async.
        """
        ...

    @abstractmethod
    def standby(self) -> None:
        """Put radio in standby mode."""
        ...

    @abstractmethod
    def sleep(self) -> None:
        """Put radio in sleep mode (lowest power)."""
        ...

    @abstractmethod
    def channel_activity_detection(self) -> bool:
        """
        Perform Channel Activity Detection.
        Returns True if channel is busy.
        """
        ...

    @abstractmethod
    def get_rssi(self) -> int:
        """Get current RSSI in dBm."""
        ...

    @abstractmethod
    def get_snr(self) -> float:
        """Get Signal-to-Noise Ratio in dB."""
        ...

    @abstractmethod
    def set_frequency(self, freq_hz: int) -> None:
        """Set operating frequency in Hz."""
        ...

    @abstractmethod
    def set_tx_power(self, power_dbm: int) -> None:
        """Set transmit power in dBm."""
        ...

    def on_receive(self, callback: Callable[[bytes, int, float], None]) -> None:
        """Register callback for received packets: callback(data, rssi, snr)."""
        self._rx_callback = callback

    def on_tx_done(self, callback: Callable[[], None]) -> None:
        """Register callback for transmission complete."""
        self._tx_done_callback = callback

    def on_cad_done(self, callback: Callable[[bool], None]) -> None:
        """Register callback for CAD result: callback(channel_busy)."""
        self._cad_callback = callback

    def get_stats(self) -> RadioStats:
        """Get radio statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset radio statistics."""
        self.stats = RadioStats()


class SX1276Driver(RadioInterface):
    """
    Driver for Semtech SX1276/SX1278 LoRa transceivers.
    
    Supports: HopeRF RFM95W, RFM96W, RFM97W, RFM98W
    Frequency: 137-1020 MHz (SX1276), 137-525 MHz (SX1278)
    Max TX Power: +20 dBm (PA_BOOST)
    Interface: SPI
    """

    # SX1276 Register Map
    REG_FIFO = 0x00
    REG_OP_MODE = 0x01
    REG_FR_MSB = 0x06
    REG_FR_MID = 0x07
    REG_FR_LSB = 0x08
    REG_PA_CONFIG = 0x09
    REG_PA_RAMP = 0x0A
    REG_OCP = 0x0B
    REG_LNA = 0x0C
    REG_FIFO_ADDR_PTR = 0x0D
    REG_FIFO_TX_BASE = 0x0E
    REG_FIFO_RX_BASE = 0x0F
    REG_FIFO_RX_CURRENT = 0x10
    REG_IRQ_FLAGS_MASK = 0x11
    REG_IRQ_FLAGS = 0x12
    REG_RX_NB_BYTES = 0x13
    REG_PKT_SNR = 0x19
    REG_PKT_RSSI = 0x1A
    REG_RSSI = 0x1B
    REG_MODEM_CONFIG_1 = 0x1D
    REG_MODEM_CONFIG_2 = 0x1E
    REG_SYMB_TIMEOUT_LSB = 0x1F
    REG_PREAMBLE_MSB = 0x20
    REG_PREAMBLE_LSB = 0x21
    REG_PAYLOAD_LENGTH = 0x22
    REG_MAX_PAYLOAD_LENGTH = 0x23
    REG_MODEM_CONFIG_3 = 0x26
    REG_FREQ_ERROR_MSB = 0x28
    REG_FREQ_ERROR_MID = 0x29
    REG_FREQ_ERROR_LSB = 0x2A
    REG_RSSI_WIDEBAND = 0x2C
    REG_DETECTION_OPTIMIZE = 0x31
    REG_INVERT_IQ = 0x33
    REG_DETECTION_THRESHOLD = 0x37
    REG_SYNC_WORD = 0x39
    REG_DIO_MAPPING_1 = 0x40
    REG_DIO_MAPPING_2 = 0x41
    REG_VERSION = 0x42
    REG_TEMP = 0x3C
    REG_PA_DAC = 0x4D

    # Operating modes
    MODE_SLEEP = 0x00
    MODE_STDBY = 0x01
    MODE_TX = 0x03
    MODE_RX_CONTINUOUS = 0x05
    MODE_RX_SINGLE = 0x06
    MODE_CAD = 0x07
    MODE_LORA = 0x80

    # IRQ flags
    IRQ_TX_DONE = 0x08
    IRQ_RX_DONE = 0x40
    IRQ_CRC_ERROR = 0x20
    IRQ_CAD_DONE = 0x04
    IRQ_CAD_DETECTED = 0x01

    def __init__(self, spi_bus: int = 0, spi_cs: int = 0,
                 reset_pin: int = 25, dio0_pin: int = 24,
                 config: Optional[RadioConfig] = None):
        super().__init__(config)
        self.spi_bus = spi_bus
        self.spi_cs = spi_cs
        self.reset_pin = reset_pin
        self.dio0_pin = dio0_pin
        self._spi = None
        self._gpio = None

    def initialize(self) -> bool:
        """Initialize SX1276 radio."""
        try:
            # Import SPI and GPIO (platform-dependent)
            self._init_hardware()

            # Reset the module
            self.reset()
            time.sleep(0.01)

            # Verify chip version
            version = self._read_register(self.REG_VERSION)
            if version != 0x12:
                logger.error(f'SX1276 version mismatch: expected 0x12, got 0x{version:02X}')
                return False

            logger.info(f'SX1276 detected (version 0x{version:02X})')

            # Set LoRa mode
            self._set_mode(self.MODE_SLEEP)
            self._write_register(self.REG_OP_MODE, self.MODE_LORA | self.MODE_SLEEP)
            time.sleep(0.01)

            # Apply configuration
            self.configure(self.config)

            # Set to standby
            self.standby()

            self.state = RadioState.STANDBY
            logger.info('SX1276 initialized successfully')
            return True

        except Exception as e:
            logger.error(f'SX1276 initialization failed: {e}')
            return False

    def _init_hardware(self):
        """Initialize SPI and GPIO interfaces."""
        try:
            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(self.spi_bus, self.spi_cs)
            self._spi.max_speed_hz = 5_000_000
            self._spi.mode = 0
        except ImportError:
            logger.warning('spidev not available - using stub SPI')
            self._spi = _StubSPI()

        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.reset_pin, GPIO.OUT)
            GPIO.setup(self.dio0_pin, GPIO.IN)
        except ImportError:
            logger.warning('RPi.GPIO not available - using stub GPIO')
            self._gpio = _StubGPIO()

    def reset(self) -> None:
        """Hardware reset via reset pin."""
        if self._gpio:
            self._gpio.output(self.reset_pin, 0)
            time.sleep(0.001)
            self._gpio.output(self.reset_pin, 1)
            time.sleep(0.01)

    def configure(self, config: RadioConfig) -> None:
        """Apply full radio configuration."""
        self.config = config
        self._set_mode(self.MODE_SLEEP)

        # Set frequency
        self.set_frequency(config.frequency)

        # Set bandwidth, coding rate, implicit header
        bw_map = {
            7800: 0, 10400: 1, 15600: 2, 20800: 3,
            31250: 4, 41700: 5, 62500: 6, 125000: 7,
            250000: 8, 500000: 9
        }
        bw_val = bw_map.get(config.bandwidth, 7)

        cr_val = config.coding_rate - 4  # CR_4_5=1, CR_4_6=2, etc.
        implicit = 1 if config.implicit_header else 0

        modem_config_1 = (bw_val << 4) | (cr_val << 1) | implicit
        self._write_register(self.REG_MODEM_CONFIG_1, modem_config_1)

        # Set spreading factor, CRC
        crc = 0x04 if config.crc_enabled else 0x00
        modem_config_2 = (config.spreading_factor << 4) | crc
        self._write_register(self.REG_MODEM_CONFIG_2, modem_config_2)

        # Set low data rate optimize and AGC
        ldro = 0x08 if config.low_data_rate_optimize else 0x00
        agc = 0x04  # AGC auto on
        self._write_register(self.REG_MODEM_CONFIG_3, ldro | agc)

        # Set TX power
        self.set_tx_power(config.tx_power)

        # Set preamble length
        self._write_register(self.REG_PREAMBLE_MSB, (config.preamble_length >> 8) & 0xFF)
        self._write_register(self.REG_PREAMBLE_LSB, config.preamble_length & 0xFF)

        # Set sync word
        self._write_register(self.REG_SYNC_WORD, config.sync_word & 0xFF)

        # Set detection optimize for SF6
        if config.spreading_factor == 6:
            self._write_register(self.REG_DETECTION_OPTIMIZE, 0x05)
            self._write_register(self.REG_DETECTION_THRESHOLD, 0x0C)
        else:
            self._write_register(self.REG_DETECTION_OPTIMIZE, 0x03)
            self._write_register(self.REG_DETECTION_THRESHOLD, 0x0A)

        # Set IQ inversion
        if config.iq_inverted:
            self._write_register(self.REG_INVERT_IQ, 0x67)
        else:
            self._write_register(self.REG_INVERT_IQ, 0x27)

        # Set over-current protection
        ocp_trim = min(27, max(0, (config.ocp_current - 45) // 5))
        self._write_register(self.REG_OCP, 0x20 | ocp_trim)

        # FIFO pointers
        self._write_register(self.REG_FIFO_TX_BASE, 0x00)
        self._write_register(self.REG_FIFO_RX_BASE, 0x00)

        logger.debug(
            f'SX1276 configured: {config.frequency/1e6:.1f}MHz, '
            f'BW={config.bandwidth}Hz, SF={config.spreading_factor}, '
            f'CR=4/{config.coding_rate}, TX={config.tx_power}dBm'
        )

    def transmit(self, data: bytes) -> bool:
        """Transmit data packet."""
        if len(data) > 255:
            logger.error(f'Payload too large: {len(data)} > 255')
            return False

        self.standby()

        # Set FIFO pointer to TX base
        self._write_register(self.REG_FIFO_ADDR_PTR, 0x00)

        # Write data to FIFO
        for byte in data:
            self._write_register(self.REG_FIFO, byte)

        # Set payload length
        self._write_register(self.REG_PAYLOAD_LENGTH, len(data))

        # Map DIO0 to TX Done
        self._write_register(self.REG_DIO_MAPPING_1, 0x40)

        # Clear IRQ flags
        self._write_register(self.REG_IRQ_FLAGS, 0xFF)

        # Start transmission
        self._set_mode(self.MODE_TX)
        self.state = RadioState.TX

        self.stats.tx_packets += 1
        self.stats.tx_bytes += len(data)

        logger.debug(f'TX started: {len(data)} bytes')
        return True

    def receive(self, timeout_ms: int = 0) -> Optional[bytes]:
        """Start receiving mode."""
        self.standby()

        # Set FIFO pointer to RX base
        self._write_register(self.REG_FIFO_ADDR_PTR, 0x00)

        # Map DIO0 to RX Done
        self._write_register(self.REG_DIO_MAPPING_1, 0x00)

        # Clear IRQ flags
        self._write_register(self.REG_IRQ_FLAGS, 0xFF)

        if timeout_ms == 0:
            # Continuous RX
            self._set_mode(self.MODE_RX_CONTINUOUS)
        else:
            # Single RX with timeout
            self._set_mode(self.MODE_RX_SINGLE)

        self.state = RadioState.RX

        if timeout_ms > 0:
            # Blocking receive with timeout
            start = time.monotonic()
            while (time.monotonic() - start) < (timeout_ms / 1000):
                irq = self._read_register(self.REG_IRQ_FLAGS)
                if irq & self.IRQ_RX_DONE:
                    if irq & self.IRQ_CRC_ERROR:
                        self.stats.crc_errors += 1
                        self._write_register(self.REG_IRQ_FLAGS, 0xFF)
                        return None

                    # Read received data
                    nb_bytes = self._read_register(self.REG_RX_NB_BYTES)
                    current_addr = self._read_register(self.REG_FIFO_RX_CURRENT)
                    self._write_register(self.REG_FIFO_ADDR_PTR, current_addr)

                    data = bytes(
                        self._read_register(self.REG_FIFO) for _ in range(nb_bytes)
                    )

                    self.stats.rx_packets += 1
                    self.stats.rx_bytes += nb_bytes
                    self.stats.last_rssi = self.get_rssi()
                    self.stats.last_snr = self.get_snr()

                    self._write_register(self.REG_IRQ_FLAGS, 0xFF)
                    return data

                time.sleep(0.001)

            return None  # Timeout

        return None  # Async mode

    def standby(self) -> None:
        """Enter standby mode."""
        self._set_mode(self.MODE_STDBY)
        self.state = RadioState.STANDBY

    def sleep(self) -> None:
        """Enter sleep mode."""
        self._set_mode(self.MODE_SLEEP)
        self.state = RadioState.SLEEP

    def channel_activity_detection(self) -> bool:
        """Perform CAD to detect channel activity."""
        self.standby()

        # Map DIO0 to CAD Done
        self._write_register(self.REG_DIO_MAPPING_1, 0x80)
        self._write_register(self.REG_IRQ_FLAGS, 0xFF)

        self._set_mode(self.MODE_CAD)
        self.state = RadioState.CAD

        # Wait for CAD to complete
        while True:
            irq = self._read_register(self.REG_IRQ_FLAGS)
            if irq & self.IRQ_CAD_DONE:
                busy = bool(irq & self.IRQ_CAD_DETECTED)
                self._write_register(self.REG_IRQ_FLAGS, 0xFF)
                if busy:
                    self.stats.channel_busy += 1
                self.standby()
                return busy
            time.sleep(0.001)

    def get_rssi(self) -> int:
        """Get last packet RSSI in dBm."""
        raw = self._read_register(self.REG_PKT_RSSI)
        if self.config.frequency < 868_000_000:
            return raw - 164
        return raw - 157

    def get_snr(self) -> float:
        """Get last packet SNR in dB."""
        raw = self._read_register(self.REG_PKT_SNR)
        if raw > 127:
            raw -= 256
        return raw / 4.0

    def set_frequency(self, freq_hz: int) -> None:
        """Set operating frequency."""
        self.config.frequency = freq_hz
        frf = int((freq_hz << 19) / 32_000_000)
        self._write_register(self.REG_FR_MSB, (frf >> 16) & 0xFF)
        self._write_register(self.REG_FR_MID, (frf >> 8) & 0xFF)
        self._write_register(self.REG_FR_LSB, frf & 0xFF)

    def set_tx_power(self, power_dbm: int) -> None:
        """Set transmit power (2-20 dBm with PA_BOOST)."""
        self.config.tx_power = max(2, min(20, power_dbm))
        if power_dbm > 17:
            # Enable +20dBm mode
            self._write_register(self.REG_PA_DAC, 0x87)
            self._write_register(self.REG_PA_CONFIG, 0x80 | (power_dbm - 5))
        else:
            self._write_register(self.REG_PA_DAC, 0x84)
            self._write_register(self.REG_PA_CONFIG, 0x80 | (power_dbm - 2))

    def _set_mode(self, mode: int) -> None:
        """Set operating mode."""
        current = self._read_register(self.REG_OP_MODE)
        self._write_register(self.REG_OP_MODE, (current & 0x80) | mode)

    def _read_register(self, address: int) -> int:
        """Read a single register via SPI."""
        response = self._spi.xfer2([address & 0x7F, 0x00])
        return response[1]

    def _write_register(self, address: int, value: int) -> None:
        """Write a single register via SPI."""
        self._spi.xfer2([address | 0x80, value])


class SX1262Driver(RadioInterface):
    """
    Driver for Semtech SX1262 LoRa transceiver.
    
    Features: Lower power, better blocking performance, 
    support for LoRa and GFSK modulation.
    Frequency: 150-960 MHz
    Max TX Power: +22 dBm
    Interface: SPI with BUSY pin
    """

    # SX1262 Commands
    CMD_SET_SLEEP = 0x84
    CMD_SET_STANDBY = 0x80
    CMD_SET_FS = 0xC1
    CMD_SET_TX = 0x83
    CMD_SET_RX = 0x82
    CMD_SET_CAD = 0xC5
    CMD_SET_PKT_TYPE = 0x8A
    CMD_SET_RF_FREQ = 0x86
    CMD_SET_PA_CONFIG = 0x95
    CMD_SET_TX_PARAMS = 0x8E
    CMD_SET_BUFFER_BASE = 0x8F
    CMD_SET_MOD_PARAMS = 0x8B
    CMD_SET_PKT_PARAMS = 0x8C
    CMD_GET_STATUS = 0xC0
    CMD_GET_RSSI = 0x15
    CMD_READ_BUFFER = 0x1E
    CMD_WRITE_BUFFER = 0x0E
    CMD_GET_IRQ_STATUS = 0x12
    CMD_CLR_IRQ_STATUS = 0x02
    CMD_SET_DIO_IRQ = 0x08
    CMD_SET_SYNC_WORD = 0x06
    CMD_GET_PKT_STATUS = 0x14

    def __init__(self, spi_bus: int = 0, spi_cs: int = 0,
                 reset_pin: int = 25, busy_pin: int = 24,
                 dio1_pin: int = 22,
                 config: Optional[RadioConfig] = None):
        super().__init__(config)
        self.spi_bus = spi_bus
        self.spi_cs = spi_cs
        self.reset_pin = reset_pin
        self.busy_pin = busy_pin
        self.dio1_pin = dio1_pin
        self._spi = None
        self._gpio = None

    def initialize(self) -> bool:
        """Initialize SX1262 radio."""
        try:
            self._init_hardware()
            self.reset()
            time.sleep(0.01)

            # Wait for BUSY to go low
            self._wait_busy()

            # Set standby mode
            self._send_command(self.CMD_SET_STANDBY, [0x00])  # STDBY_RC

            # Set packet type to LoRa
            self._send_command(self.CMD_SET_PKT_TYPE, [0x01])  # PACKET_TYPE_LORA

            # Apply configuration
            self.configure(self.config)

            self.state = RadioState.STANDBY
            logger.info('SX1262 initialized successfully')
            return True

        except Exception as e:
            logger.error(f'SX1262 initialization failed: {e}')
            return False

    def _init_hardware(self):
        """Initialize SPI and GPIO."""
        try:
            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(self.spi_bus, self.spi_cs)
            self._spi.max_speed_hz = 8_000_000
            self._spi.mode = 0
        except ImportError:
            self._spi = _StubSPI()

        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.reset_pin, GPIO.OUT)
            GPIO.setup(self.busy_pin, GPIO.IN)
            GPIO.setup(self.dio1_pin, GPIO.IN)
        except ImportError:
            self._gpio = _StubGPIO()

    def reset(self) -> None:
        """Hardware reset."""
        if self._gpio:
            self._gpio.output(self.reset_pin, 0)
            time.sleep(0.001)
            self._gpio.output(self.reset_pin, 1)
            time.sleep(0.01)

    def configure(self, config: RadioConfig) -> None:
        """Apply radio configuration."""
        self.config = config

        # Set frequency
        self.set_frequency(config.frequency)

        # Set PA config
        self._send_command(self.CMD_SET_PA_CONFIG, [
            0x04,  # paDutyCycle
            0x07,  # hpMax
            0x00,  # deviceSel (SX1262)
            0x01   # paLut
        ])

        # Set TX power
        self.set_tx_power(config.tx_power)

        # Set modulation parameters
        bw_map = {
            7800: 0x00, 10400: 0x08, 15600: 0x01, 20800: 0x09,
            31250: 0x02, 41700: 0x0A, 62500: 0x03, 125000: 0x04,
            250000: 0x05, 500000: 0x06
        }
        bw_val = bw_map.get(config.bandwidth, 0x04)
        ldro = 0x01 if config.low_data_rate_optimize else 0x00

        self._send_command(self.CMD_SET_MOD_PARAMS, [
            config.spreading_factor,
            bw_val,
            config.coding_rate - 4,
            ldro
        ])

        # Set packet parameters
        crc_type = 0x01 if config.crc_enabled else 0x00
        header_type = 0x01 if config.implicit_header else 0x00
        iq = 0x01 if config.iq_inverted else 0x00

        self._send_command(self.CMD_SET_PKT_PARAMS, [
            (config.preamble_length >> 8) & 0xFF,
            config.preamble_length & 0xFF,
            header_type,
            255,  # Max payload length
            crc_type,
            iq
        ])

        # Set buffer base addresses
        self._send_command(self.CMD_SET_BUFFER_BASE, [0x00, 0x00])

        # Set sync word (LoRa public: 0x3444, private: 0x1424)
        sw = config.sync_word
        self._write_register_sx(0x0740, [(sw >> 8) & 0xFF, sw & 0xFF])

        logger.debug(f'SX1262 configured: {config.frequency/1e6:.1f}MHz')

    def transmit(self, data: bytes) -> bool:
        """Transmit data packet."""
        if len(data) > 255:
            return False

        self.standby()

        # Write data to buffer
        self._send_command(self.CMD_WRITE_BUFFER, [0x00] + list(data))

        # Set payload length in packet params
        self._send_command(self.CMD_SET_PKT_PARAMS, [
            (self.config.preamble_length >> 8) & 0xFF,
            self.config.preamble_length & 0xFF,
            0x01 if self.config.implicit_header else 0x00,
            len(data),
            0x01 if self.config.crc_enabled else 0x00,
            0x01 if self.config.iq_inverted else 0x00
        ])

        # Clear IRQ
        self._send_command(self.CMD_CLR_IRQ_STATUS, [0xFF, 0xFF])

        # Set TX with no timeout
        self._send_command(self.CMD_SET_TX, [0x00, 0x00, 0x00])

        self.state = RadioState.TX
        self.stats.tx_packets += 1
        self.stats.tx_bytes += len(data)

        return True

    def receive(self, timeout_ms: int = 0) -> Optional[bytes]:
        """Start receiving."""
        self.standby()
        self._send_command(self.CMD_CLR_IRQ_STATUS, [0xFF, 0xFF])

        if timeout_ms == 0:
            # Continuous RX
            self._send_command(self.CMD_SET_RX, [0xFF, 0xFF, 0xFF])
        else:
            # Timed RX
            timeout_val = int(timeout_ms * 15.625)  # 64µs steps
            self._send_command(self.CMD_SET_RX, [
                (timeout_val >> 16) & 0xFF,
                (timeout_val >> 8) & 0xFF,
                timeout_val & 0xFF
            ])

        self.state = RadioState.RX
        return None  # Async — use callback

    def standby(self) -> None:
        self._send_command(self.CMD_SET_STANDBY, [0x00])
        self.state = RadioState.STANDBY

    def sleep(self) -> None:
        self._send_command(self.CMD_SET_SLEEP, [0x00])
        self.state = RadioState.SLEEP

    def channel_activity_detection(self) -> bool:
        self.standby()
        self._send_command(self.CMD_CLR_IRQ_STATUS, [0xFF, 0xFF])
        self._send_command(self.CMD_SET_CAD, [])
        self.state = RadioState.CAD

        # Wait for CAD done
        while True:
            status = self._send_command(self.CMD_GET_IRQ_STATUS, [0x00, 0x00])
            if status and len(status) >= 3:
                irq = (status[1] << 8) | status[2]
                if irq & 0x0080:  # CadDone
                    busy = bool(irq & 0x0040)  # CadDetected
                    self._send_command(self.CMD_CLR_IRQ_STATUS, [0xFF, 0xFF])
                    if busy:
                        self.stats.channel_busy += 1
                    self.standby()
                    return busy
            time.sleep(0.001)

    def get_rssi(self) -> int:
        result = self._send_command(self.CMD_GET_RSSI, [0x00, 0x00])
        if result and len(result) >= 2:
            return -(result[1] // 2)
        return -120

    def get_snr(self) -> float:
        result = self._send_command(self.CMD_GET_PKT_STATUS, [0x00, 0x00, 0x00, 0x00])
        if result and len(result) >= 3:
            snr_raw = result[2]
            if snr_raw > 127:
                snr_raw -= 256
            return snr_raw / 4.0
        return 0.0

    def set_frequency(self, freq_hz: int) -> None:
        self.config.frequency = freq_hz
        frf = int((freq_hz * (2 ** 25)) / 32_000_000)
        self._send_command(self.CMD_SET_RF_FREQ, [
            (frf >> 24) & 0xFF,
            (frf >> 16) & 0xFF,
            (frf >> 8) & 0xFF,
            frf & 0xFF
        ])

    def set_tx_power(self, power_dbm: int) -> None:
        self.config.tx_power = max(-9, min(22, power_dbm))
        # Ramp time: 200µs
        self._send_command(self.CMD_SET_TX_PARAMS, [
            power_dbm & 0xFF,
            0x04  # Ramp 200µs
        ])

    def _wait_busy(self, timeout: float = 1.0):
        """Wait for BUSY pin to go low."""
        if hasattr(self._gpio, 'input'):
            start = time.monotonic()
            while self._gpio.input(self.busy_pin):
                if time.monotonic() - start > timeout:
                    raise TimeoutError('SX1262 BUSY timeout')
                time.sleep(0.0001)

    def _send_command(self, cmd: int, args: list = None) -> Optional[list]:
        """Send SPI command to SX1262."""
        self._wait_busy()
        payload = [cmd] + (args or [])
        result = self._spi.xfer2(payload)
        return result[1:] if len(result) > 1 else None

    def _write_register_sx(self, address: int, data: list):
        """Write register(s) on SX1262."""
        self._wait_busy()
        cmd = [0x0D, (address >> 8) & 0xFF, address & 0xFF] + data
        self._spi.xfer2(cmd)


# ============================================================
# Stub implementations for development/testing without hardware
# ============================================================

class _StubSPI:
    """Stub SPI interface for testing without hardware."""
    _registers = {}

    def open(self, bus, cs):
        pass

    def xfer2(self, data):
        if len(data) >= 2:
            addr = data[0] & 0x7F
            if data[0] & 0x80:  # Write
                self._registers[addr] = data[1]
            else:  # Read
                val = self._registers.get(addr, 0x12 if addr == 0x42 else 0x00)
                return [0x00, val]
        return [0x00] * len(data)

    @property
    def max_speed_hz(self):
        return 0

    @max_speed_hz.setter
    def max_speed_hz(self, val):
        pass

    @property
    def mode(self):
        return 0

    @mode.setter
    def mode(self, val):
        pass


class _StubGPIO:
    """Stub GPIO interface for testing without hardware."""
    BCM = 11
    OUT = 0
    IN = 1
    _state = {}

    @classmethod
    def setmode(cls, mode):
        pass

    @classmethod
    def setup(cls, pin, direction):
        cls._state[pin] = 0

    @classmethod
    def output(cls, pin, value):
        cls._state[pin] = value

    @classmethod
    def input(cls, pin):
        return 0  # Not busy

    @classmethod
    def cleanup(cls):
        cls._state.clear()
