//! High-performance BrianProtocol message codec.
//!
//! Wire format:
//! Magic(2B) + Ver(1B) + Type(1B) + Prio(1B) + Seq(4B) + TS(8B) +
//! SrcLen(1B) + Source + PayloadLen(4B) + Payload + HMAC(32B)

use ring::hmac;
use bytes::{Buf, BufMut, BytesMut};
use thiserror::Error;

const BRIAN_MAGIC: [u8; 2] = [0x42, 0x52]; // "BR"
const BRIAN_VERSION: u8 = 1;
const HMAC_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    EmergencyStop = 0,
    SafetyStatus = 1,
    JointCommand = 2,
    JointState = 3,
    PerceptionFrame = 4,
    WorldState = 5,
    TaskCommand = 6,
    Heartbeat = 7,
    ConfigUpdate = 8,
    Telemetry = 9,
    Ack = 10,
}

impl TryFrom<u8> for MessageType {
    type Error = CodecError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::EmergencyStop),
            1 => Ok(Self::SafetyStatus),
            2 => Ok(Self::JointCommand),
            3 => Ok(Self::JointState),
            4 => Ok(Self::PerceptionFrame),
            5 => Ok(Self::WorldState),
            6 => Ok(Self::TaskCommand),
            7 => Ok(Self::Heartbeat),
            8 => Ok(Self::ConfigUpdate),
            9 => Ok(Self::Telemetry),
            10 => Ok(Self::Ack),
            _ => Err(CodecError::InvalidMessageType(value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BrianMessage {
    pub msg_type: MessageType,
    pub priority: u8,
    pub sequence_num: u32,
    pub timestamp_ns: u64,
    pub source_id: String,
    pub payload: Vec<u8>,
    pub hmac_signature: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Invalid magic bytes")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),
    #[error("Invalid message type: {0}")]
    InvalidMessageType(u8),
    #[error("HMAC verification failed")]
    HmacFailed,
    #[error("Buffer too short: need {need} bytes, got {got}")]
    BufferTooShort { need: usize, got: usize },
    #[error("Source ID too long: {0} bytes")]
    SourceIdTooLong(usize),
}

pub struct ProtocolCodec {
    hmac_key: hmac::Key,
}

impl ProtocolCodec {
    pub fn new(key: &[u8]) -> Self {
        Self {
            hmac_key: hmac::Key::new(hmac::HMAC_SHA256, key),
        }
    }

    /// Encode a message to wire format.
    pub fn encode(&self, msg: &BrianMessage) -> Result<Vec<u8>, CodecError> {
        let src_bytes = msg.source_id.as_bytes();
        if src_bytes.len() > 255 {
            return Err(CodecError::SourceIdTooLong(src_bytes.len()));
        }

        let mut buf = BytesMut::with_capacity(
            2 + 1 + 1 + 1 + 4 + 8 + 1 + src_bytes.len() + 4 + msg.payload.len() + HMAC_SIZE
        );

        // Header
        buf.put_slice(&BRIAN_MAGIC);
        buf.put_u8(BRIAN_VERSION);
        buf.put_u8(msg.msg_type as u8);
        buf.put_u8(msg.priority);
        buf.put_u32(msg.sequence_num);
        buf.put_u64(msg.timestamp_ns);
        buf.put_u8(src_bytes.len() as u8);
        buf.put_slice(src_bytes);
        buf.put_u32(msg.payload.len() as u32);
        buf.put_slice(&msg.payload);

        // Compute HMAC over everything before the signature
        let tag = hmac::sign(&self.hmac_key, &buf);
        buf.put_slice(tag.as_ref());

        Ok(buf.to_vec())
    }

    /// Decode a message from wire format.
    pub fn decode(&self, data: &[u8]) -> Result<BrianMessage, CodecError> {
        if data.len() < 2 + 1 + 1 + 1 + 4 + 8 + 1 + 4 + HMAC_SIZE {
            return Err(CodecError::BufferTooShort {
                need: 22 + HMAC_SIZE,
                got: data.len(),
            });
        }

        let mut buf = &data[..];

        // Magic
        let magic = [buf[0], buf[1]];
        buf = &buf[2..];
        if magic != BRIAN_MAGIC {
            return Err(CodecError::InvalidMagic);
        }

        // Version
        let version = buf[0];
        buf = &buf[1..];
        if version != BRIAN_VERSION {
            return Err(CodecError::UnsupportedVersion(version));
        }

        // Type, Priority, Sequence, Timestamp
        let msg_type = MessageType::try_from(buf[0])?;
        let priority = buf[1];
        buf = &buf[2..];

        let mut seq_bytes = [0u8; 4];
        seq_bytes.copy_from_slice(&buf[..4]);
        let sequence_num = u32::from_be_bytes(seq_bytes);
        buf = &buf[4..];

        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&buf[..8]);
        let timestamp_ns = u64::from_be_bytes(ts_bytes);
        buf = &buf[8..];

        // Source ID
        let src_len = buf[0] as usize;
        buf = &buf[1..];
        let source_id = String::from_utf8_lossy(&buf[..src_len]).to_string();
        buf = &buf[src_len..];

        // Payload
        let mut plen_bytes = [0u8; 4];
        plen_bytes.copy_from_slice(&buf[..4]);
        let payload_len = u32::from_be_bytes(plen_bytes) as usize;
        buf = &buf[4..];
        let payload = buf[..payload_len].to_vec();
        buf = &buf[payload_len..];

        // HMAC
        let hmac_signature = buf[..HMAC_SIZE].to_vec();

        // Verify HMAC
        let msg_data = &data[..data.len() - HMAC_SIZE];
        hmac::verify(&self.hmac_key, msg_data, &hmac_signature)
            .map_err(|_| CodecError::HmacFailed)?;

        Ok(BrianMessage {
            msg_type,
            priority,
            sequence_num,
            timestamp_ns,
            source_id,
            payload,
            hmac_signature,
        })
    }
}
