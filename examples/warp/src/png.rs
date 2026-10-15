//! PNG files of 8-bit RGB pixels, written without dependencies.

/// A PNG of 8-bit RGB pixels, with stored (uncompressed) deflate blocks: no dependencies.
pub fn encode(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
    let crc_table: Vec<u32> = (0..256u32)
        .map(|n| {
            (0..8).fold(n, |c, _| {
                if c & 1 == 1 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                }
            })
        })
        .collect();
    let crc = |bytes: &[u8]| {
        !bytes.iter().fold(!0u32, |c, b| {
            crc_table[((c ^ u32::from(*b)) & 0xff) as usize] ^ (c >> 8)
        })
    };
    let chunk = |out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = ty.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    };
    let mut raw = Vec::with_capacity(((w * 3 + 1) * h) as usize);
    for line in rgb.chunks_exact(w as usize * 3) {
        raw.push(0); // no filter
        raw.extend_from_slice(line);
    }
    let mut z = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, c) in chunks.iter().enumerate() {
        z.push(u8::from(i + 1 == chunks.len()));
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    // Adler-32.
    let (a, b) = raw.iter().fold((1u32, 0u32), |(a, b), &x| {
        let a = (a + u32::from(x)) % 65521;
        (a, (b + a) % 65521)
    });
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    /// The signature, the header's size and kind, and the closing chunk with its known CRC.
    #[test]
    fn a_png_is_well_formed() {
        let png = super::encode(2, 1, &[255, 0, 0, 0, 255, 0]);
        assert_eq!(png[..8], [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(png[12..16], *b"IHDR");
        assert_eq!(png[16..24], [0, 0, 0, 2, 0, 0, 0, 1]);
        let end = &png[png.len() - 12..];
        assert_eq!(
            end,
            [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82]
        );
    }
}
