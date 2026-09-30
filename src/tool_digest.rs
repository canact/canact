//! SHA-256 digest of a caller tool list.
//!
//! Canonical bytes are a compact JSON array. Each object is encoded with
//! keys in the order `name`, `description`, `parameters`.

#![cfg_attr(not(any(test, feature = "runtime")), allow(dead_code))]

pub(crate) struct DigestTool<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters: &'a serde_json::Value,
}

pub(crate) fn tools_digest(tools: &[DigestTool<'_>]) -> [u8; 32] {
    sha256(&canonical_tools_bytes(tools))
}

pub(crate) fn tools_digest_hex(tools: &[DigestTool<'_>]) -> String {
    hex32(&tools_digest(tools))
}

#[cfg(feature = "runtime")]
fn probe_rows(tools: &[crate::client::ProbeTool]) -> Vec<DigestTool<'_>> {
    tools
        .iter()
        .map(|tool| DigestTool {
            name: tool.name.as_str(),
            description: tool.description.as_str(),
            parameters: &tool.parameters,
        })
        .collect()
}

#[cfg(feature = "runtime")]
pub fn probe_tools_digest(tools: &[crate::client::ProbeTool]) -> [u8; 32] {
    tools_digest(&probe_rows(tools))
}

#[cfg(feature = "runtime")]
pub(crate) fn probe_tools_digest_hex(tools: &[crate::client::ProbeTool]) -> String {
    tools_digest_hex(&probe_rows(tools))
}

pub(crate) fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn canonical_tools_bytes(tools: &[DigestTool<'_>]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(b'[');
    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            buf.push(b',');
        }
        let object = serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        });
        let name = serde_json::to_string(&object["name"]).expect("tool name json");
        let description =
            serde_json::to_string(&object["description"]).expect("tool description json");
        let parameters =
            serde_json::to_string(&object["parameters"]).expect("tool parameters json");
        buf.extend_from_slice(b"{\"name\":");
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(b",\"description\":");
        buf.extend_from_slice(description.as_bytes());
        buf.extend_from_slice(b",\"parameters\":");
        buf.extend_from_slice(parameters.as_bytes());
        buf.push(b'}');
    }
    buf.push(b']');
    buf
}

#[allow(clippy::unreadable_literal)]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[allow(clippy::unreadable_literal)]
fn sha256(data: &[u8]) -> [u8; 32] {
    let bit_len = u64::try_from(data.len())
        .expect("tool list byte length fits in u64")
        .wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    for chunk in msg.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in schedule.iter_mut().enumerate().take(16) {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let earlier = schedule[index - 15];
            let recent = schedule[index - 2];
            let small = earlier.rotate_right(7) ^ earlier.rotate_right(18) ^ (earlier >> 3);
            let big = recent.rotate_right(17) ^ recent.rotate_right(19) ^ (recent >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(big);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];
        for (round_constant, word) in SHA256_K.into_iter().zip(schedule) {
            let upper = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(upper)
                .wrapping_add(choose)
                .wrapping_add(round_constant)
                .wrapping_add(word);
            let lower = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = lower.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{DigestTool, sha256, tools_digest, tools_digest_hex};

    #[test]
    fn sha256_abc_matches_known_vector() {
        let abc = sha256(b"abc");
        assert_eq!(
            hex_of(&abc),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let empty = sha256(b"");
        assert_eq!(
            hex_of(&empty),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let parameters = serde_json::json!({"type": "object"});
        let tools = [DigestTool {
            name: "read",
            description: "Read a file",
            parameters: &parameters,
        }];
        let hex = tools_digest_hex(&tools);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, hex_of(&tools_digest(&tools)));
        assert!(hex.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    fn hex_of(bytes: &[u8; 32]) -> String {
        super::hex32(bytes)
    }
}
