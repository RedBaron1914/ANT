use std::fs::OpenOptions;
use std::path::Path;
use memmap2::MmapMut;
use super::tensor::Tensor1D;

pub struct DiskKVMemory {
    pub capacity: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    mmap: MmapMut,
    pub current_size: usize,
    pub write_cursor: usize,
}

impl DiskKVMemory {
    pub const HEADER_SIZE: usize = 48; // 6 * 8 bytes

    pub fn new<P: AsRef<Path>>(path: P, capacity: usize, key_dim: usize, val_dim: usize) -> std::io::Result<Self> {
        let file_size = Self::HEADER_SIZE + (capacity * key_dim * 4) + (capacity * val_dim * 4) + (capacity * 8);
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
            
        // Set file size if it's new or needs expansion
        if file.metadata()?.len() < file_size as u64 {
            file.set_len(file_size as u64)?;
        }
        
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Initialize or read header
        let mut current_size: usize = 0;
        let mut write_cursor: usize = 0;
        let is_new = mmap[0..8].iter().all(|&b| b == 0);
        
        if is_new {
            let version = 3u64;
            mmap[0..8].copy_from_slice(&version.to_le_bytes());
            mmap[8..16].copy_from_slice(&capacity.to_le_bytes());
            mmap[16..24].copy_from_slice(&current_size.to_le_bytes());
            mmap[24..32].copy_from_slice(&key_dim.to_le_bytes());
            mmap[32..40].copy_from_slice(&val_dim.to_le_bytes());
            mmap[40..48].copy_from_slice(&write_cursor.to_le_bytes());
        } else {
            let file_version = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
            assert_eq!(file_version, 3, "Unsupported version in .ant file! Expected version 3.");
            let file_capacity = usize::from_le_bytes(mmap[8..16].try_into().unwrap());
            current_size = usize::from_le_bytes(mmap[16..24].try_into().unwrap());
            let file_key_dim = usize::from_le_bytes(mmap[24..32].try_into().unwrap());
            let file_val_dim = usize::from_le_bytes(mmap[32..40].try_into().unwrap());
            write_cursor = usize::from_le_bytes(mmap[40..48].try_into().unwrap());
            
            assert_eq!(file_capacity, capacity, "Capacity mismatch");
            assert_eq!(file_key_dim, key_dim, "Key dim mismatch");
            assert_eq!(file_val_dim, val_dim, "Val dim mismatch");
        }
        
        Ok(Self {
            capacity,
            key_dim,
            val_dim,
            mmap,
            current_size,
            write_cursor,
        })
    }

    pub fn open_existing<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
            
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        if mmap.len() < Self::HEADER_SIZE {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Cartridge file smaller than header"));
        }
        
        let file_version = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if file_version != 3 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unsupported version {} in cartridge! Expected 3", file_version)));
        }
        
        let capacity = usize::from_le_bytes(mmap[8..16].try_into().unwrap());
        let current_size = usize::from_le_bytes(mmap[16..24].try_into().unwrap());
        let key_dim = usize::from_le_bytes(mmap[24..32].try_into().unwrap());
        let val_dim = usize::from_le_bytes(mmap[32..40].try_into().unwrap());
        let write_cursor = usize::from_le_bytes(mmap[40..48].try_into().unwrap());

        Ok(Self {
            capacity,
            key_dim,
            val_dim,
            mmap,
            current_size,
            write_cursor,
        })
    }

    fn normalize_inplace(tensor: &mut Tensor1D) {
        let mut norm_sq = 0.0;
        for val in tensor.data.iter() {
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-8 {
            for val in tensor.data.iter_mut() {
                *val /= norm;
            }
        }
    }

    fn set_current_size(&mut self, size: usize) {
        self.current_size = size;
        self.mmap[16..24].copy_from_slice(&size.to_le_bytes());
    }

    fn set_write_cursor(&mut self, cursor: usize) {
        self.write_cursor = cursor;
        self.mmap[40..48].copy_from_slice(&cursor.to_le_bytes());
    }

    pub fn get_key(&self, index: usize) -> &[f32] {
        let start = Self::HEADER_SIZE + index * self.key_dim * 4;
        let end = start + self.key_dim * 4;
        bytemuck::cast_slice(&self.mmap[start..end])
    }

    pub fn get_key_mut(&mut self, index: usize) -> &mut [f32] {
        let start = Self::HEADER_SIZE + index * self.key_dim * 4;
        let end = start + self.key_dim * 4;
        bytemuck::cast_slice_mut(&mut self.mmap[start..end])
    }

    pub fn get_val(&self, index: usize) -> &[f32] {
        let keys_byte_size = self.capacity * self.key_dim * 4;
        let start = Self::HEADER_SIZE + keys_byte_size + index * self.val_dim * 4;
        let end = start + self.val_dim * 4;
        bytemuck::cast_slice(&self.mmap[start..end])
    }

    pub fn get_val_mut(&mut self, index: usize) -> &mut [f32] {
        let keys_byte_size = self.capacity * self.key_dim * 4;
        let start = Self::HEADER_SIZE + keys_byte_size + index * self.val_dim * 4;
        let end = start + self.val_dim * 4;
        bytemuck::cast_slice_mut(&mut self.mmap[start..end])
    }

    pub fn get_metadata(&self, index: usize) -> u64 {
        let keys_byte_size = self.capacity * self.key_dim * 4;
        let vals_byte_size = self.capacity * self.val_dim * 4;
        let start = Self::HEADER_SIZE + keys_byte_size + vals_byte_size + index * 8;
        u64::from_le_bytes(self.mmap[start..start + 8].try_into().unwrap())
    }

    pub fn set_metadata(&mut self, index: usize, val: u64) {
        let keys_byte_size = self.capacity * self.key_dim * 4;
        let vals_byte_size = self.capacity * self.val_dim * 4;
        let start = Self::HEADER_SIZE + keys_byte_size + vals_byte_size + index * 8;
        self.mmap[start..start + 8].copy_from_slice(&val.to_le_bytes());
    }

    pub fn add_memory(&mut self, mut key: Tensor1D, value: Tensor1D, metadata: u64) {
        Self::normalize_inplace(&mut key);
        
        let index = self.write_cursor;
        self.get_key_mut(index).copy_from_slice(&key.data);
        self.get_val_mut(index).copy_from_slice(&value.data);
        self.set_metadata(index, metadata);

        self.set_write_cursor((self.write_cursor + 1) % self.capacity);
        if self.current_size < self.capacity {
            self.set_current_size(self.current_size + 1);
        }
    }

    pub fn lookup(&self, query: &Tensor1D, query_metadata: u64, top_k: usize) -> (Tensor1D, Vec<usize>) {
        let size = std::cmp::min(self.current_size, self.capacity);
        if size == 0 {
            return (Tensor1D::new(self.val_dim), vec![]);
        }

        let mut q_norm = query.clone();
        Self::normalize_inplace(&mut q_norm);

        let query_polarity = query_metadata & 1;

        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(size);
        for i in 0..size {
            let key = self.get_key(i);
            let mut dot = 0.0;
            for j in 0..self.key_dim {
                dot += q_norm.data[j] * key[j];
            }
            let polarity = self.get_metadata(i) & 1;
            if query_polarity != polarity {
                dot -= 0.5; // Penalty for opposite polarity
            }
            scores.push((i, dot));
        }

        let actual_k = std::cmp::min(top_k, scores.len());
        
        if actual_k < scores.len() {
            scores.select_nth_unstable_by(actual_k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let top_scores = &mut scores[0..actual_k];
        top_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut output = Tensor1D::new(self.val_dim);
        let mut top_indices = Vec::with_capacity(actual_k);

        for i in 0..actual_k {
            let score = top_scores[i].1;
            let val_idx = top_scores[i].0;
            top_indices.push(val_idx);
            let val = self.get_val(val_idx);
            let weight = if score > 0.0 { score } else { 0.0 };
            for j in 0..self.val_dim {
                output.data[j] += val[j] * weight;
            }
        }

        (output, top_indices)
    }

    /// Reward-Modulated Hebbian Learning (Online Adaptation)
    pub fn update_memory(&mut self, query: &Tensor1D, reward: f32, learning_rate: f32) {
        let size = std::cmp::min(self.current_size, self.capacity);
        if size == 0 || reward.abs() < 1e-5 { return; }

        let mut q_norm = query.clone();
        Self::normalize_inplace(&mut q_norm);

        let mut best_idx = 0;
        let mut max_dot = -f32::INFINITY;
        
        for i in 0..size {
            let key = self.get_key(i);
            let mut dot = 0.0;
            for j in 0..self.key_dim {
                dot += q_norm.data[j] * key[j];
            }
            if dot > max_dot {
                max_dot = dot;
                best_idx = i;
            }
        }
        
        if max_dot > 0.0 {
            let key_dim_val = self.key_dim;
            let key_slice = self.get_key_mut(best_idx);
            for j in 0..key_dim_val {
                key_slice[j] += learning_rate * reward * q_norm.data[j];
            }
            
            let mut key_tensor = Tensor1D::new(key_dim_val);
            key_tensor.data.copy_from_slice(key_slice);
            Self::normalize_inplace(&mut key_tensor);
            key_slice.copy_from_slice(&key_tensor.data);
        }
    }

    pub fn compress_and_prune(&mut self, similarity_threshold: f32) {
        let size = std::cmp::min(self.current_size, self.capacity);
        if size < 2 { return; }

        let mut valid_indices = vec![true; size];
        let mut new_keys = Vec::new();
        let mut new_vals = Vec::new();
        let mut new_metas = Vec::new();

        for i in 0..size {
            if !valid_indices[i] { continue; }
            
            let original_key = self.get_key(i).to_vec(); 
            let original_meta = self.get_metadata(i);
            
            let mut merged_key = original_key.clone();
            let mut merged_val = self.get_val(i).to_vec();
            let mut merge_count = 1.0f32;

            for j in (i + 1)..size {
                if !valid_indices[j] { continue; }
                let key_j = self.get_key(j);
                
                let mut dot = 0.0;
                for d in 0..self.key_dim { dot += original_key[d] * key_j[d]; }
                
                if dot > similarity_threshold {
                    for d in 0..self.key_dim { merged_key[d] += key_j[d]; }
                    for d in 0..self.val_dim { merged_val[d] += self.get_val(j)[d]; }
                    merge_count += 1.0;
                    valid_indices[j] = false;
                }
            }

            let mut norm_sq = 0.0;
            for d in 0..self.key_dim { 
                merged_key[d] /= merge_count; 
                norm_sq += merged_key[d] * merged_key[d];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-8 {
                for d in 0..self.key_dim { merged_key[d] /= norm; }
            }

            for d in 0..self.val_dim { merged_val[d] /= merge_count; }

            new_keys.push(merged_key);
            new_vals.push(merged_val);
            new_metas.push(original_meta);
        }

        let new_size = new_keys.len();
        for i in 0..new_size {
            self.get_key_mut(i).copy_from_slice(&new_keys[i]);
            self.get_val_mut(i).copy_from_slice(&new_vals[i]);
            self.set_metadata(i, new_metas[i]);
        }

        self.set_current_size(new_size);
        self.set_write_cursor(new_size % self.capacity);
        println!("Memory pruned: {} entries reduced to {}", size, new_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_disk_kv_memory() {
        let path = "test_memory.ant";
        let _ = fs::remove_file(path); 
        
        {
            let mut memory = DiskKVMemory::new(path, 10, 3, 2).unwrap();
            
            let mut key1 = Tensor1D::new(3); key1.data = vec![1.0, 0.0, 0.0];
            let mut val1 = Tensor1D::new(2); val1.data = vec![10.0, -10.0];
            memory.add_memory(key1, val1, 0);
            
            let mut key2 = Tensor1D::new(3); key2.data = vec![0.0, 1.0, 0.0];
            let mut val2 = Tensor1D::new(2); val2.data = vec![5.0, 5.0];
            memory.add_memory(key2, val2, 0);
            
            let mut query = Tensor1D::new(3); query.data = vec![0.8, 0.6, 0.0];
            let (_out, indices) = memory.lookup(&query, 0, 1);
            
            assert_eq!(indices[0], 0);
            
            memory.update_memory(&query, -0.5, 0.1);
        } 
        
        {
            let memory = DiskKVMemory::new(path, 10, 3, 2).unwrap();
            assert_eq!(memory.current_size, 2);
            
            let mut query = Tensor1D::new(3); query.data = vec![0.8, 0.6, 0.0];
            let (out, _) = memory.lookup(&query, 0, 1);
            
            assert!(out.data[0] < 8.0); 
        }
        
        let _ = fs::remove_file(path);
    }
}
