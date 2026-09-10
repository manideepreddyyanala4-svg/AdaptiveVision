"""A minimal Modbus TCP server standing in for a line PLC.

Lets the station's reject handshake be exercised end-to-end with no real
PLC on the bench: it accepts Modbus TCP connections, serves a small coil and
holding-register file, and logs every read and write so a coil flip (the
pneumatic rejector firing on a FAIL verdict) is visible on the console.

Implements only the four function codes the station uses -- read coils (0x01),
read holding registers (0x03), write single coil (0x05), and write multiple
registers (0x10) -- which is what ``adaptivevision.communication.ModbusClient``
declares. It is a bench simulator, not a conformant Modbus implementation.

Usage:
    python scripts/mock_plc.py [--host 127.0.0.1] [--port 5020]
"""

from __future__ import annotations

import argparse
import logging
import socketserver
import struct
import sys

logger = logging.getLogger("mock_plc")

#: Size of the simulated coil and holding-register files.
COIL_COUNT = 256
REGISTER_COUNT = 256

#: Modbus function codes this simulator answers.
_READ_COILS = 0x01
_READ_HOLDING = 0x03
_WRITE_COIL = 0x05
_WRITE_REGISTERS = 0x10

#: Modbus exception code for an unsupported function.
_ILLEGAL_FUNCTION = 0x01

#: Value a "write single coil" request uses to mean ON.
_COIL_ON = 0xFF00

#: MBAP header: transaction id, protocol id, length, unit id.
_MBAP_STRUCT = struct.Struct(">HHHB")
_MBAP_LEN = _MBAP_STRUCT.size


class PlcState:
    """The simulated PLC's coil and register files."""

    def __init__(self) -> None:
        """Initialize all coils off and all registers zero."""
        self.coils = [False] * COIL_COUNT
        self.registers = [0] * REGISTER_COUNT

    def read_coils(self, address: int, count: int) -> list[bool]:
        """Return ``count`` coils starting at ``address``."""
        return self.coils[address : address + count]

    def read_registers(self, address: int, count: int) -> list[int]:
        """Return ``count`` holding registers starting at ``address``."""
        return self.registers[address : address + count]

    def write_coil(self, address: int, value: bool) -> None:
        """Set a single coil, logging the reject line specially."""
        self.coils[address] = value
        state = "ON" if value else "OFF"
        if value:
            logger.warning("coil[%d] <- %s  ** REJECT ASSERTED (pneumatic push) **", address, state)
        else:
            logger.info("coil[%d] <- %s", address, state)

    def write_registers(self, address: int, values: list[int]) -> None:
        """Write consecutive holding registers."""
        self.registers[address : address + len(values)] = values
        logger.info("registers[%d..%d] <- %s", address, address + len(values) - 1, values)


def _pack_bits(bits: list[bool]) -> bytes:
    """Pack coil states into Modbus's little-endian bit layout."""
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def build_response(state: PlcState, unit: int, pdu: bytes) -> bytes:
    """Build the response PDU for one request PDU.

    Args:
        state: The simulated PLC state to read from or mutate.
        unit: Modbus unit identifier (echoed; this simulator serves all units).
        pdu: The request protocol data unit, function code first.

    Returns:
        The response PDU, which may be a Modbus exception response.
    """
    del unit
    function = pdu[0]

    if function == _READ_COILS:
        address, count = struct.unpack(">HH", pdu[1:5])
        payload = _pack_bits(state.read_coils(address, count))
        logger.info("read coils[%d..%d]", address, address + count - 1)
        return bytes([function, len(payload)]) + payload

    if function == _READ_HOLDING:
        address, count = struct.unpack(">HH", pdu[1:5])
        values = state.read_registers(address, count)
        payload = b"".join(struct.pack(">H", v) for v in values)
        logger.info("read registers[%d..%d]", address, address + count - 1)
        return bytes([function, len(payload)]) + payload

    if function == _WRITE_COIL:
        address, raw = struct.unpack(">HH", pdu[1:5])
        state.write_coil(address, raw == _COIL_ON)
        return pdu[:5]

    if function == _WRITE_REGISTERS:
        address, count = struct.unpack(">HH", pdu[1:5])
        values = list(struct.unpack(">" + "H" * count, pdu[6 : 6 + count * 2]))
        state.write_registers(address, values)
        return struct.pack(">BHH", function, address, count)

    logger.warning("unsupported function code 0x%02X", function)
    return bytes([function | 0x80, _ILLEGAL_FUNCTION])


class _Handler(socketserver.BaseRequestHandler):
    """Serves Modbus TCP frames for one client connection."""

    def handle(self) -> None:
        """Read requests until the client disconnects."""
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        logger.info("client connected: %s", peer)
        state: PlcState = self.server.state  # type: ignore[attr-defined]
        try:
            while True:
                header = self._read_exactly(_MBAP_LEN)
                if header is None:
                    break
                transaction, protocol, length, unit = _MBAP_STRUCT.unpack(header)
                pdu = self._read_exactly(length - 1)
                if pdu is None:
                    break
                response = build_response(state, unit, pdu)
                self.request.sendall(
                    _MBAP_STRUCT.pack(transaction, protocol, len(response) + 1, unit) + response
                )
        except (ConnectionError, OSError) as exc:
            logger.info("client %s dropped: %s", peer, exc)
        finally:
            logger.info("client disconnected: %s", peer)

    def _read_exactly(self, count: int) -> bytes | None:
        """Read exactly ``count`` bytes, or ``None`` if the peer closed."""
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.request.recv(count - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)


class MockPlcServer(socketserver.ThreadingTCPServer):
    """A threaded Modbus TCP server backed by a single :class:`PlcState`."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int) -> None:
        """Bind the server and initialize its PLC state."""
        super().__init__((host, port), _Handler)
        self.state = PlcState()


def main(argv: list[str] | None = None) -> int:
    """Run the simulator until interrupted.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on clean shutdown).
    """
    parser = argparse.ArgumentParser(description="Modbus TCP PLC simulator")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=5020, help="bind port")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    server = MockPlcServer(args.host, args.port)
    logger.info("mock PLC listening on %s:%d (%d coils, %d registers)",
                args.host, args.port, COIL_COUNT, REGISTER_COUNT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
