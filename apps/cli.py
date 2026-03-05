"""
NullSec LoRa Mesh - Command Line Interface

CLI tool for managing LoRa mesh nodes, sending messages,
and monitoring network status.

Usage:
    nullsec-mesh node start --id 0x01 --freq 915.0
    nullsec-mesh send --to 0x02 "Hello mesh!"
    nullsec-mesh status
    nullsec-mesh neighbors
    nullsec-mesh routes
    nullsec-mesh monitor
    nullsec-mesh benchmark --target 0x02
"""

import os
import sys
import time
import signal
import struct
import threading
import json
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Ensure our parent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protocol import (
    MeshFrame, MessageType, MAX_PAYLOAD_SIZE, PROTOCOL_VERSION,
)
from protocol.crypto import CryptoEngine
from protocol.compression import Compressor, CompressionMode
from protocol.fec import ReedSolomon
from protocol.routing import MeshRouter
from node import MeshNode, LoRaConfig, ReceivedMessage
from radio.hal import RadioConfig, Bandwidth, SpreadingFactor, CodingRate
from radio.channel import ChannelManager, Region, AdaptiveDataRate
from transport.reliable import TransportLayer

console = Console() if HAS_RICH else None


def _print(msg, style=None):
    """Print with or without rich formatting."""
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


# ── CLI Application ──

@click.group()
@click.version_option(version="0.2.0-alpha", prog_name="nullsec-mesh")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """NullSec LoRa Mesh — Zero-leakage encrypted mesh networking."""
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.option("--id", "node_id", required=True, type=str, help="Node ID (hex, e.g., 0x01)")
@click.option("--freq", default=915.0, type=float, help="Frequency in MHz")
@click.option("--sf", default=7, type=click.IntRange(6, 12), help="Spreading factor (6-12)")
@click.option("--bw", default=125000, type=int, help="Bandwidth in Hz")
@click.option("--power", default=14, type=click.IntRange(2, 22), help="TX power in dBm")
@click.option("--region", default="us915", type=click.Choice(["us915", "eu868"]), help="Regulatory region")
@click.option("--compression", default="adaptive", type=click.Choice(["adaptive", "lz4", "zstd", "none"]))
@click.option("--fec", default=16, type=int, help="FEC parity symbols (0=disabled)")
@click.option("--no-encrypt", is_flag=True, help="Disable encryption")
@click.option("--config", type=click.Path(), help="Config file path (JSON)")
def start(node_id, freq, sf, bw, power, region, compression, fec, no_encrypt, config):
    """Start a mesh node."""
    # Parse node ID
    nid = int(node_id, 0) if isinstance(node_id, str) else node_id

    # Load config file if provided
    if config:
        with open(config) as f:
            cfg = json.load(f)
            freq = cfg.get("frequency", freq)
            sf = cfg.get("spreading_factor", sf)
            bw = cfg.get("bandwidth", bw)
            power = cfg.get("tx_power", power)
            region = cfg.get("region", region)

    lora_config = LoRaConfig(
        frequency=freq,
        bandwidth=bw,
        spreading_factor=sf,
        tx_power=power,
    )

    _print(Panel.fit(
        f"[bold green]NullSec LoRa Mesh Node[/bold green]\n"
        f"Node ID: [cyan]0x{nid:08X}[/cyan]\n"
        f"Frequency: [yellow]{freq} MHz[/yellow]\n"
        f"SF: [yellow]{sf}[/yellow] | BW: [yellow]{bw/1000:.0f} kHz[/yellow]\n"
        f"TX Power: [yellow]{power} dBm[/yellow]\n"
        f"Region: [yellow]{region.upper()}[/yellow]\n"
        f"Encryption: [{'green' if not no_encrypt else 'red'}]"
        f"{'ON' if not no_encrypt else 'OFF'}[/]\n"
        f"Compression: [yellow]{compression}[/yellow]\n"
        f"FEC: [yellow]{fec} symbols[/yellow]",
        title="⚡ Node Configuration",
        border_style="green",
    )) if console else print(f"Starting node 0x{nid:08X} on {freq}MHz")

    # Create and start node
    node = MeshNode(
        node_id=nid,
        config=lora_config,
        compression=compression,
        encryption=not no_encrypt,
        fec_symbols=fec,
    )

    # Register callbacks
    node.on_message(lambda msg: _on_message(msg))
    node.on_neighbor(lambda nid, rssi: _on_neighbor(nid, rssi))

    node.start()

    # Store node reference for other commands
    _save_node_state(nid, freq, sf, bw, power, region)

    _print("[bold green]✓ Node started[/bold green]")
    _print("[dim]Press Ctrl+C to stop[/dim]")

    # Main loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _print("\n[yellow]Shutting down...[/yellow]")
        node.stop()
        _print("[green]✓ Node stopped[/green]")


@cli.command()
@click.option("--to", "dest", required=True, type=str, help="Destination node ID (hex)")
@click.argument("message")
@click.option("--unreliable", is_flag=True, help="Don't wait for ACK")
def send(dest, message, unreliable):
    """Send a message to a node."""
    dst = int(dest, 0)

    state = _load_node_state()
    if not state:
        _print("[red]Error: No active node. Run 'start' first.[/red]")
        return

    nid = state["node_id"]
    _print(
        f"[cyan]→[/cyan] Sending to [bold]0x{dst:08X}[/bold]: "
        f"[green]{message}[/green] "
        f"({'unreliable' if unreliable else 'reliable'})"
    ) if console else print(f"Sending to 0x{dst:08X}: {message}")

    # In a full implementation, this would connect to the running node
    # via IPC (unix socket, shared memory, etc.)
    # For now, demonstrate the protocol stack
    node = MeshNode(node_id=nid, encryption=True)
    success = node.send(dst, message.encode(), reliable=not unreliable)

    if success:
        _print("[green]✓ Message queued[/green]")
    else:
        _print("[red]✗ Send failed (no route?)[/red]")


@cli.command()
def status():
    """Show node and network status."""
    state = _load_node_state()
    if not state:
        _print("[red]No active node.[/red]")
        return

    if not console:
        print(f"Node: 0x{state['node_id']:08X}")
        print(f"Freq: {state['frequency']}MHz, SF{state['sf']}")
        return

    table = Table(title="🔧 Node Status", border_style="cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Node ID", f"0x{state['node_id']:08X}")
    table.add_row("Frequency", f"{state['frequency']} MHz")
    table.add_row("Spreading Factor", f"SF{state['sf']}")
    table.add_row("Bandwidth", f"{state['bw']/1000:.0f} kHz")
    table.add_row("TX Power", f"{state['power']} dBm")
    table.add_row("Region", state['region'].upper())
    table.add_row("Protocol Version", str(PROTOCOL_VERSION))
    table.add_row("Max Payload", f"{MAX_PAYLOAD_SIZE} bytes")
    table.add_row("Uptime", _format_uptime(state.get('started', time.time())))

    console.print(table)


@cli.command()
def neighbors():
    """Show discovered neighbors."""
    if not console:
        print("Neighbor discovery requires an active node")
        return

    table = Table(title="📡 Neighbors", border_style="blue")
    table.add_column("Node ID", style="cyan")
    table.add_column("RSSI", style="yellow")
    table.add_column("SNR", style="yellow")
    table.add_column("Last Seen", style="green")
    table.add_column("Link Quality", style="magenta")

    # In production, would query the running node via IPC
    console.print(table)
    console.print("[dim]Start a node to discover neighbors[/dim]")


@cli.command()
def routes():
    """Show routing table."""
    if not console:
        print("Routing table requires an active node")
        return

    table = Table(title="🗺️  Routing Table", border_style="yellow")
    table.add_column("Destination", style="cyan")
    table.add_column("Next Hop", style="green")
    table.add_column("Hops", style="yellow", justify="right")
    table.add_column("Seq #", style="blue", justify="right")
    table.add_column("Lifetime", style="magenta")
    table.add_column("Valid", style="green")

    console.print(table)
    console.print("[dim]Routes are discovered on-demand via RREQ/RREP[/dim]")


@cli.command()
@click.option("--duration", default=0, type=int, help="Monitor duration in seconds (0=forever)")
def monitor(duration):
    """Monitor mesh traffic in real-time."""
    _print("[bold cyan]📻 Mesh Traffic Monitor[/bold cyan]")
    _print("[dim]Listening for mesh frames...[/dim]")
    _print()

    start_time = time.time()

    try:
        while True:
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    _print("\n[yellow]Monitor stopped[/yellow]")


@cli.command()
@click.option("--target", required=True, type=str, help="Target node ID")
@click.option("--count", default=10, type=int, help="Number of packets")
@click.option("--size", default=32, type=int, help="Payload size in bytes")
def benchmark(target, count, size):
    """Run a link quality benchmark."""
    dst = int(target, 0)

    _print(f"[bold]📊 Benchmark: 0x{dst:08X}[/bold]")
    _print(f"Packets: {count}, Size: {size} bytes")
    _print()

    # Demonstrate protocol stack calculations
    config = RadioConfig()

    table = Table(title="Protocol Stack Analysis", border_style="green")
    table.add_column("Layer", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Overhead", justify="right", style="yellow")

    raw_size = size
    table.add_row("Application Data", f"{raw_size}B", "—")

    # Compression estimate
    comp = Compressor(CompressionMode.ADAPTIVE)
    compressed = comp.compress(os.urandom(size))
    comp_size = len(compressed)
    table.add_row("+ Compression Header", f"{comp_size}B", f"+3B")

    # Encryption
    enc_overhead = 16  # auth tag
    table.add_row("+ Encryption (AEAD)", f"{comp_size + enc_overhead}B",
                   f"+{enc_overhead}B")

    # Transport header
    transport_overhead = 12
    table.add_row("+ Transport Header", f"{comp_size + enc_overhead + transport_overhead}B",
                   f"+{transport_overhead}B")

    # Mesh frame header
    frame_overhead = 18
    total = comp_size + enc_overhead + transport_overhead + frame_overhead
    table.add_row("+ Mesh Frame Header", f"{total}B", f"+{frame_overhead}B")

    # FEC
    fec = ReedSolomon(nsym=16)
    fec_total = total + 16
    table.add_row("+ FEC (RS-16)", f"{fec_total}B", f"+16B")

    _print(table) if console else print(f"Total: {fec_total}B")

    # Time on air
    for sf_val in [7, 9, 12]:
        config_test = RadioConfig(spreading_factor=SpreadingFactor(sf_val))
        toa = config_test.time_on_air(fec_total)
        _print(f"  SF{sf_val}: Time on Air = [cyan]{toa*1000:.1f}ms[/cyan], "
               f"Rate = [green]{config_test.bit_rate:.0f} bps[/green]")


@cli.command()
def keygen():
    """Generate a new identity keypair."""
    crypto = CryptoEngine()
    pubkey = crypto.public_key_bytes.hex()

    _print("[bold green]🔑 New Identity Generated[/bold green]")
    _print(f"Public Key: [cyan]{pubkey}[/cyan]")
    _print("[dim]Private key stored in memory (not exported for security)[/dim]")


@cli.command()
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
def info(fmt):
    """Show protocol and capability information."""
    info_data = {
        "protocol_version": PROTOCOL_VERSION,
        "max_payload": MAX_PAYLOAD_SIZE,
        "max_frame": 255,
        "header_size": 18,
        "encryption": "ChaCha20-Poly1305 AEAD",
        "key_exchange": "X25519 ECDH",
        "key_derivation": "HKDF-SHA256",
        "compression": ["None", "LZ4 Fast", "Zstandard Balanced", "Zstandard Max"],
        "fec": "Reed-Solomon GF(2^8)",
        "routing": "AODV (Ad-hoc On-Demand Distance Vector)",
        "channel_access": "CSMA/CA with LBT",
        "frequency_hopping": "PRNG-seeded pseudo-random hopping",
        "radios_supported": ["SX1276/SX1278 (Sub-GHz)", "SX1262 (Sub-GHz v2)"],
        "spreading_factors": "SF6-SF12",
        "bandwidths": "7.8-500 kHz",
        "anti_replay": "Sliding window (128-bit bitmap)",
        "key_rotation": "Automatic (1000 msgs or 1 hour)",
    }

    if fmt == "json":
        _print(json.dumps(info_data, indent=2))
        return

    if not console:
        for k, v in info_data.items():
            print(f"{k}: {v}")
        return

    table = Table(title="🛡️  NullSec LoRa Mesh Protocol", border_style="cyan")
    table.add_column("Feature", style="bold")
    table.add_column("Details", style="green")

    for key, val in info_data.items():
        display_key = key.replace("_", " ").title()
        if isinstance(val, list):
            val = ", ".join(val)
        table.add_row(display_key, str(val))

    console.print(table)


@cli.command()
@click.option("--region", default="us915", type=click.Choice(["us915", "eu868"]))
def channels(region):
    """Show available channels for a region."""
    reg = Region.US_915 if region == "us915" else Region.EU_868
    mgr = ChannelManager(region=reg, hop_enabled=False)

    if not console:
        for ch in mgr.plan.channels:
            print(f"{ch/1e6:.1f} MHz")
        return

    table = Table(title=f"📻 Channel Plan: {region.upper()}", border_style="yellow")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Frequency", style="cyan")
    table.add_column("Duty Cycle Limit", style="yellow")
    table.add_column("Max TX Power", style="green")

    for i, freq in enumerate(mgr.plan.channels):
        table.add_row(
            str(i + 1),
            f"{freq/1e6:.3f} MHz",
            f"{mgr.plan.duty_cycle*100:.0f}%",
            f"{mgr.plan.max_tx_power} dBm",
        )

    console.print(table)
    console.print(f"\n[dim]Total channels: {len(mgr.plan.channels)}[/dim]")


# ── Helpers ──

def _on_message(msg: ReceivedMessage):
    """Callback for received messages."""
    _print(
        f"[green]←[/green] [bold]0x{msg.src:08X}[/bold] "
        f"({msg.rssi}dBm, {msg.snr:.1f}dB): "
        f"[cyan]{msg.data.decode('utf-8', errors='replace')}[/cyan]"
    )


def _on_neighbor(node_id: int, rssi: int):
    """Callback for neighbor discovery."""
    _print(
        f"[blue]📡[/blue] Neighbor: [bold]0x{node_id:08X}[/bold] "
        f"(RSSI: {rssi}dBm)"
    )


def _format_uptime(start_time: float) -> str:
    """Format uptime as human-readable string."""
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    elif elapsed < 3600:
        return f"{elapsed/60:.0f}m {elapsed%60:.0f}s"
    else:
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        return f"{h}h {m}m"


def _save_node_state(nid, freq, sf, bw, power, region):
    """Save node state to temp file for inter-command communication."""
    state = {
        "node_id": nid,
        "frequency": freq,
        "sf": sf,
        "bw": bw,
        "power": power,
        "region": region,
        "started": time.time(),
    }
    state_file = Path("/tmp/.nullsec-mesh-state.json")
    state_file.write_text(json.dumps(state))


def _load_node_state() -> Optional[dict]:
    """Load saved node state."""
    state_file = Path("/tmp/.nullsec-mesh-state.json")
    if state_file.exists():
        return json.loads(state_file.read_text())
    return None


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
