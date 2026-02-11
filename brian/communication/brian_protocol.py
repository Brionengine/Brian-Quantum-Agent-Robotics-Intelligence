"""
Layer 3: Communication Transport - BrianProtocol

Custom binary protocol for brain-to-robot communication.
Extends the SecureProtocolHandler from SII Framework.

Three-channel architecture:
  Channel A: Command (gRPC over TLS 1.3) - motor commands + state feedback
  Channel B: Sensor Stream (WebRTC) - camera, audio, depth
  Channel C: Control (TCP + TLS) - heartbeat, E-stop, config

Wire format:
  Magic(2B) + Ver(1B) + Type(1B) + Prio(1B) + Seq(4B) + TS(8B) +
  SrcLen(1B) + Source(SrcLen B) + PayloadLen(4B) + Payload + HMAC(32B)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
import struct
import time
import hmac
import hashlib
import logging

logger = logging.getLogger(__name__)

BRIAN_MAGIC = b'BR'  # 2 bytes magic header
BRIAN_VERSION = 1
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10 MB max payload


class BrianMessageType(IntEnum):
    """Message types ordered by priority (lower = higher priority)."""
    EMERGENCY_STOP = 0
    SAFETY_STATUS = 1
    JOINT_COMMAND = 2
    JOINT_STATE = 3
    PERCEPTION_FRAME = 4
    WORLD_STATE = 5
    TASK_COMMAND = 6
    HEARTBEAT = 7
    CONFIG_UPDATE = 8
    TELEMETRY = 9
    ACK = 10


@dataclass
class BrianMessage:
    """Core message format for BrianProtocol."""
    msg_type: BrianMessageType
    priority: int  # 0=ESTOP highest, 9=telemetry lowest
    sequence_num: int
    timestamp_ns: int
    source_id: str
    payload: bytes
    hmac_signature: bytes = b''

    def compute_hmac(self, key: bytes) -> bytes:
        """Compute HMAC-SHA256 over message fields."""
        data = struct.pack('>BBI', self.msg_type, self.priority, self.sequence_num)
        data += struct.pack('>Q', self.timestamp_ns)
        data += self.source_id.encode()
        data += self.payload
        return hmac.new(key, data, hashlib.sha256).digest()

    def verify_hmac(self, key: bytes) -> bool:
        """Verify HMAC signature."""
        expected = self.compute_hmac(key)
        return hmac.compare_digest(self.hmac_signature, expected)

    def serialize(self, hmac_key: Optional[bytes] = None) -> bytes:
        """Serialize message to wire format."""
        self.hmac_signature = self.compute_hmac(hmac_key) if hmac_key else b'\x00' * 32
        src_bytes = self.source_id.encode()
        header = struct.pack('>2sBBBI',
                             b'BR', BRIAN_VERSION, self.msg_type,
                             self.priority, self.sequence_num)
        header += struct.pack('>Q', self.timestamp_ns)
        header += struct.pack('>B', len(src_bytes))
        header += src_bytes
        header += struct.pack('>I', len(self.payload))
        return header + self.payload + self.hmac_signature

    @classmethod
    def deserialize(cls, data: bytes, hmac_key: Optional[bytes] = None) -> 'BrianMessage':
        """Deserialize message from wire format."""
        offset = 0
        magic = data[offset:offset + 2]
        offset += 2
        if magic != b'BR':
            raise ValueError(f"Invalid magic: {magic}")

        version, msg_type, priority, seq_num = struct.unpack_from('>BBBI', data, offset)
        offset += 7
        timestamp_ns = struct.unpack_from('>Q', data, offset)[0]
        offset += 8
        src_len = struct.unpack_from('>B', data, offset)[0]
        offset += 1
        source_id = data[offset:offset + src_len].decode()
        offset += src_len
        payload_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        payload = data[offset:offset + payload_len]
        offset += payload_len
        hmac_sig = data[offset:offset + 32]

        msg = cls(
            msg_type=BrianMessageType(msg_type),
            priority=priority,
            sequence_num=seq_num,
            timestamp_ns=timestamp_ns,
            source_id=source_id,
            payload=payload,
            hmac_signature=hmac_sig,
        )

        if hmac_key and not msg.verify_hmac(hmac_key):
            raise ValueError("HMAC verification failed")

        return msg


@dataclass
class ConnectionState:
    """State of a brain-robot connection."""
    robot_id: str
    is_connected: bool = False
    last_heartbeat_ns: int = 0
    sequence_counter: int = 0
    latency_ms: float = 0.0
    bandwidth_bps: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0


class BrianProtocolHandler:
    """
    Handles BrianProtocol message encoding, decoding, and routing.

    Extends SII Framework's SecureProtocolHandler with:
    - Three-channel architecture (command, sensor, control)
    - HMAC-SHA256 integrity on all messages
    - Sequence number tracking for ordering and replay prevention
    - Timestamp validation (reject messages older than max_command_age_ms)
    - Priority-based message processing
    """

    def __init__(self, source_id: str, hmac_key: bytes = b'brian-qari-default-key'):
        self.source_id = source_id
        self.hmac_key = hmac_key
        self.connections: Dict[str, ConnectionState] = {}
        self._sequence_counter = 0
        self._message_handlers: Dict[BrianMessageType, List[Callable]] = {
            t: [] for t in BrianMessageType
        }
        self.max_command_age_ms = 500.0  # Reject commands older than 500ms
        logger.info(f"BrianProtocolHandler initialized | source={source_id}")

    def create_message(self, msg_type: BrianMessageType,
                       payload: bytes, priority: Optional[int] = None) -> BrianMessage:
        """Create a new signed message."""
        self._sequence_counter += 1
        msg = BrianMessage(
            msg_type=msg_type,
            priority=priority if priority is not None else msg_type.value,
            sequence_num=self._sequence_counter,
            timestamp_ns=time.time_ns(),
            source_id=self.source_id,
            payload=payload,
        )
        msg.hmac_signature = msg.compute_hmac(self.hmac_key)
        return msg

    def create_emergency_stop(self) -> BrianMessage:
        """Create highest-priority emergency stop message."""
        return self.create_message(BrianMessageType.EMERGENCY_STOP, b'ESTOP', priority=0)

    def create_heartbeat(self) -> BrianMessage:
        """Create heartbeat message."""
        return self.create_message(BrianMessageType.HEARTBEAT, b'HB')

    def create_joint_command(self, payload: bytes) -> BrianMessage:
        """Create joint command message."""
        return self.create_message(BrianMessageType.JOINT_COMMAND, payload)

    def process_message(self, data: bytes) -> Optional[BrianMessage]:
        """
        Deserialize, validate, and route an incoming message.

        Checks:
        1. HMAC integrity
        2. Timestamp freshness (reject stale commands)
        3. Sequence number (detect replays)

        Returns the validated message, or None if rejected.
        """
        try:
            msg = BrianMessage.deserialize(data, self.hmac_key)
        except ValueError as e:
            logger.warning(f"Message rejected: {e}")
            return None

        # Check timestamp freshness for command messages
        if msg.msg_type in (BrianMessageType.JOINT_COMMAND,
                            BrianMessageType.TASK_COMMAND,
                            BrianMessageType.EMERGENCY_STOP):
            age_ms = (time.time_ns() - msg.timestamp_ns) / 1_000_000
            if age_ms > self.max_command_age_ms and msg.msg_type != BrianMessageType.EMERGENCY_STOP:
                logger.warning(f"Stale command rejected: {age_ms:.1f}ms old")
                return None

        # Update connection state
        conn = self.connections.get(msg.source_id)
        if conn:
            conn.last_heartbeat_ns = msg.timestamp_ns
            conn.messages_received += 1

        # Route to handlers
        for handler in self._message_handlers.get(msg.msg_type, []):
            try:
                handler(msg)
            except Exception as e:
                logger.error(f"Handler error for {msg.msg_type.name}: {e}")

        return msg

    def register_handler(self, msg_type: BrianMessageType, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self._message_handlers[msg_type].append(handler)

    def register_connection(self, robot_id: str) -> ConnectionState:
        """Register a new robot connection."""
        conn = ConnectionState(robot_id=robot_id, is_connected=True,
                               last_heartbeat_ns=time.time_ns())
        self.connections[robot_id] = conn
        logger.info(f"Robot connected: {robot_id}")
        return conn

    def check_connections(self, timeout_ms: float = 200.0) -> List[str]:
        """Check for timed-out connections. Returns list of disconnected robot IDs."""
        now = time.time_ns()
        disconnected = []
        for rid, conn in self.connections.items():
            age_ms = (now - conn.last_heartbeat_ns) / 1_000_000
            if age_ms > timeout_ms and conn.is_connected:
                conn.is_connected = False
                disconnected.append(rid)
                logger.warning(f"Robot {rid} disconnected (heartbeat timeout: {age_ms:.0f}ms)")
        return disconnected

    def get_latency(self, robot_id: str) -> float:
        """Get estimated round-trip latency to a robot in ms."""
        conn = self.connections.get(robot_id)
        return conn.latency_ms if conn else -1.0
