use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AntHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub vocab_size: u32,
    pub embed_dim: u32,
    pub hidden_size: u32,
    pub memory_capacity: u32,
    pub reserved: [u32; 8],
}

unsafe impl Zeroable for AntHeader {}
unsafe impl Pod for AntHeader {}

impl AntHeader {
    pub fn new(vocab: usize, embed: usize, hidden: usize, mem_cap: usize) -> Self {
        Self {
            magic: *b"ANT\0",
            version: 1,
            vocab_size: vocab as u32,
            embed_dim: embed as u32,
            hidden_size: hidden as u32,
            memory_capacity: mem_cap as u32,
            reserved: [0; 8],
        }
    }

    pub fn read_from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut header = Self::zeroed();
        file.read_exact(bytemuck::bytes_of_mut(&mut header))?;
        Ok(header)
    }
}

pub struct AntModel {
    pub header: AntHeader,
}
