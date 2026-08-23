use rand::Rng;
use faer::{MatRef, MatMut};

#[derive(Clone, Debug)]
pub struct Tensor1D {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Tensor1D {
    pub fn new(size: usize) -> Self {
        Self { 
            data: vec![0.0; size],
            grad: vec![0.0; size],
        }
    }
    
    pub fn from_vec(data: Vec<f32>) -> Self {
        let size = data.len();
        Self { 
            data,
            grad: vec![0.0; size],
        }
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }

    pub fn randomize(&mut self, min: f32, max: f32) {
        let mut rng = rand::thread_rng();
        for val in self.data.iter_mut() {
            *val = rng.gen_range(min..=max);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Tensor2D {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { 
            rows, 
            cols, 
            data: vec![0.0; rows * cols],
            grad: vec![0.0; rows * cols],
        }
    }
    
    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }
    
    pub fn randomize(&mut self, min: f32, max: f32) {
        let mut rng = rand::thread_rng();
        for val in self.data.iter_mut() {
            *val = rng.gen_range(min..=max);
        }
    }

    // faer integrations
    pub fn as_mat(&self) -> MatRef<'_, f32> {
        MatRef::from_row_major_slice(&self.data, self.rows, self.cols)
    }

    pub fn as_mut_mat(&mut self) -> MatMut<'_, f32> {
        MatMut::from_row_major_slice_mut(&mut self.data, self.rows, self.cols)
    }

    pub fn grad_as_mat(&self) -> MatRef<'_, f32> {
        MatRef::from_row_major_slice(&self.grad, self.rows, self.cols)
    }

    pub fn grad_as_mut_mat(&mut self) -> MatMut<'_, f32> {
        MatMut::from_row_major_slice_mut(&mut self.grad, self.rows, self.cols)
    }

    // out = input * self^T
    // out: (batch, rows) = input: (batch, cols) * self^T: (cols, rows)
    pub fn matmul_batch(&self, input: &BatchTensor, out: &mut BatchTensor) {
        let size = self.rows * self.cols * input.data.rows;
        let parallel = if size > 2_000_000 {
            faer::Par::Rayon(std::num::NonZeroUsize::new(rayon::current_num_threads()).unwrap())
        } else {
            faer::Par::Seq
        };
        faer::linalg::matmul::matmul(
            out.data.as_mut(),
            faer::Accum::Replace,
            input.data.as_ref(),
            self.as_mat().transpose(),
            1.0,
            parallel,
        );
    }

    // d_input += grad_output * self
    // d_self += grad_output^T * input
    pub fn matmul_batch_backward(&mut self, input: &mut BatchTensor, grad_output: &BatchTensor) {
        let size = self.rows * self.cols * input.data.rows;
        let parallel = if size > 2_000_000 {
            faer::Par::Rayon(std::num::NonZeroUsize::new(rayon::current_num_threads()).unwrap())
        } else {
            faer::Par::Seq
        };
        // dx += d_out * W
        faer::linalg::matmul::matmul(
            input.grad.as_mut(),
            faer::Accum::Add,
            grad_output.grad.as_ref(),
            self.as_mat(),
            1.0,
            parallel,
        );
        // dW += d_out^T * X
        faer::linalg::matmul::matmul(
            self.grad_as_mut_mat(),
            faer::Accum::Add,
            grad_output.grad.as_ref().transpose(),
            input.data.as_ref(),
            1.0,
            parallel,
        );
    }
}

#[derive(Clone, Debug)]
pub struct BatchMat {
    pub rows: usize,
    pub cols: usize,
    pub storage: Vec<f32>,
}

impl BatchMat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            storage: vec![0.0; rows * cols],
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows, "Rows mismatch in copy_from");
        assert_eq!(self.cols, other.cols, "Cols mismatch in copy_from");
        self.storage.copy_from_slice(&other.storage);
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }

    pub fn ncols(&self) -> usize {
        self.cols
    }

    pub fn as_ref(&self) -> faer::MatRef<'_, f32> {
        faer::MatRef::from_row_major_slice(&self.storage, self.rows, self.cols)
    }

    pub fn as_mut(&mut self) -> faer::MatMut<'_, f32> {
        faer::MatMut::from_row_major_slice_mut(&mut self.storage, self.rows, self.cols)
    }
}

pub trait MatExt {
    fn read(&self, i: usize, j: usize) -> f32;
    fn write(&mut self, i: usize, j: usize, val: f32);
    fn fill(&mut self, val: f32);
}

impl MatExt for BatchMat {
    #[inline]
    fn read(&self, i: usize, j: usize) -> f32 {
        self.storage[i * self.cols + j]
    }

    #[inline]
    fn write(&mut self, i: usize, j: usize, val: f32) {
        self.storage[i * self.cols + j] = val;
    }

    #[inline]
    fn fill(&mut self, val: f32) {
        self.storage.fill(val);
    }
}

#[derive(Clone, Debug)]
pub struct BatchTensor {
    pub data: BatchMat, // (batch_size, dim)
    pub grad: BatchMat, // (batch_size, dim)
}

impl BatchTensor {
    pub fn new(batch_size: usize, dim: usize) -> Self {
        Self {
            data: BatchMat::new(batch_size, dim),
            grad: BatchMat::new(batch_size, dim),
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}
