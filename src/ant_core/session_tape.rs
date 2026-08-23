use std::time::SystemTime;

#[derive(Clone, Debug)]
pub struct SessionTapeEntry {
    pub token_id: usize,
    pub timestamp: u64,
}

pub struct SessionTape {
    pub capacity: usize,
    pub tape: Vec<SessionTapeEntry>,
}

impl SessionTape {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tape: Vec::with_capacity(capacity),
        }
    }

    pub fn append(&mut self, token_id: usize) {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if self.tape.len() >= self.capacity {
            self.tape.remove(0);
        }
        self.tape.push(SessionTapeEntry { token_id, timestamp });
    }

    pub fn get_recent_tokens(&self, count: usize) -> Vec<usize> {
        let start = self.tape.len().saturating_sub(count);
        self.tape[start..].iter().map(|entry| entry.token_id).collect()
    }
}

pub struct LocalFifoBuffer {
    pub capacity: usize,
    pub buffer: Vec<usize>,
}

impl LocalFifoBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, token_id: usize) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(token_id);
    }

    pub fn get_tokens(&self) -> &[usize] {
        &self.buffer
    }
}
