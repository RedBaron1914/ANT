extern "C" __global__ void memory_lookup(
    const float* query,
    const float* keys,
    float* scores,
    int num_keys,
    int key_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        float dot = 0.0f;
        int key_offset = idx * key_dim;
        for (int i = 0; i < key_dim; ++i) {
            dot += query[i] * keys[key_offset + i];
        }
        scores[idx] = dot;
    }
}

extern "C" __global__ void zero_buffer_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

extern "C" __global__ void memory_lookup_batch(
    const float* queries,
    const float* keys,
    float* scores,
    int batch_size,
    int num_keys,
    int key_dim
) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && idx < num_keys) {
        float dot = 0.0f;
        int q_offset = b * key_dim;
        int key_offset = idx * key_dim;
        for (int i = 0; i < key_dim; ++i) {
            dot += queries[q_offset + i] * keys[key_offset + i];
        }
        scores[b * num_keys + idx] = dot;
    }
}

extern "C" __global__ void add_matrices_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

extern "C" __global__ void add_bias_kernel(
    float* data,
    const float* bias,
    int rows,
    int cols
) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        data[r * cols + c] += bias[c];
    }
}

extern "C" __global__ void bias_backward_kernel(
    const float* d_h_proj,
    float* d_b_proj,
    int rows,
    int cols
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; ++r) {
            sum += d_h_proj[r * cols + c];
        }
        d_b_proj[c] += sum;
    }
}

extern "C" __global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int* targets,
    float* d_logits,
    float* losses,
    int batch_size,
    int vocab_size,
    float norm_factor
) {
    int b = blockIdx.x; 
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    int offset = b * vocab_size;
    int target = targets[b];

    __shared__ float sdata[256];

    float local_max = -1e30f;
    for (int j = tid; j < vocab_size; j += blockDim.x) {
        float val = logits[offset + j];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < vocab_size; j += blockDim.x) {
        local_sum += expf(logits[offset + j] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    if (tid == 0) {
        float target_logit = logits[offset + target];
        losses[b] = logf(sum_exp + 1e-9f) - (target_logit - max_val);
    }

    for (int j = tid; j < vocab_size; j += blockDim.x) {
        float p = expf(logits[offset + j] - max_val) / (sum_exp + 1e-9f);
        float target_val = (j == target) ? 1.0f : 0.0f;
        d_logits[offset + j] = (p - target_val) * norm_factor;
    }
}

// -----------------------------------------------------
// NEW KERNELS FOR BATCHED BPTT IN VRAM
// -----------------------------------------------------

extern "C" __global__ void gru_forward_rz_kernel(
    const float* temp_r_in, const float* temp_r_hid, const float* b_r,
    const float* temp_z_in, const float* temp_z_hid, const float* b_z,
    float* r_out, float* z_out, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int idx = b * cols + i;
        float r_val = temp_r_in[idx] + temp_r_hid[idx] + b_r[i];
        float z_val = temp_z_in[idx] + temp_z_hid[idx] + b_z[i];
        r_out[idx] = 1.0f / (1.0f + expf(-r_val));
        z_out[idx] = 1.0f / (1.0f + expf(-z_val));
    }
}

extern "C" __global__ void gru_forward_n_h_kernel(
    const float* temp_n_in, const float* temp_n_hid, const float* b_n,
    const float* r, const float* z, const float* prev_h, const float* update_mask,
    float* n_out, float* h_out, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int idx = b * cols + i;
        float n_val = temp_n_in[idx] + b_n[i] + r[idx] * temp_n_hid[idx];
        float n_tanh = tanhf(n_val);
        n_out[idx] = n_tanh;
        
        float candidate = (1.0f - z[idx]) * n_tanh + z[idx] * prev_h[idx];
        float mask_val = update_mask[idx];
        h_out[idx] = mask_val * candidate + (1.0f - mask_val) * prev_h[idx];
    }
}

extern "C" __global__ void sparse_gating_forward_kernel(
    float* hidden, const float* b1, float* pre_activation, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int idx = b * cols + i;
        float val = hidden[idx] + b1[i];
        pre_activation[idx] = val;
        hidden[idx] = val > 0.0f ? val : 0.0f;
    }
}

extern "C" __global__ void sparse_gating_backward_kernel(
    const float* grad_output, const float* pre_activation, 
    float* d_pre, float* b1_grad, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int idx = b * cols + i;
        float val = pre_activation[idx];
        float d_relu = val > 0.0f ? 1.0f : 0.0f;
        float d_val = grad_output[idx] * d_relu;
        d_pre[idx] = d_val;
        
        // Use atomicAdd since we sum across the batch dimension
        float g = d_val;
        if (!isnan(g) && !isinf(g)) {
            atomicAdd(&b1_grad[i], g);
        }
    }
}

extern "C" __global__ void gru_backward_gates_kernel(
    const float* grad_h, const float* update_mask, const float* z, 
    const float* prev_h, const float* n, const float* temp_n_in, const float* b_n,
    const float* r, const float* temp_n_hid,
    const float* temp_r_in, const float* temp_r_hid, const float* b_r,
    const float* temp_z_in, const float* temp_z_hid, const float* b_z,
    float* grad_prev_h, float* d_n_pre, float* d_r_pre, float* d_z_pre,
    float* grad_temp_n_hid, float* b_n_grad, float* b_r_grad, float* b_z_grad,
    int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) {
        int idx = b * cols + i;
        float dh = grad_h[idx];
        float mask_val = update_mask[idx];
        float dh_gated = dh * mask_val;
        
        float z_val_cache = z[idx];
        float dn = dh_gated * (1.0f - z_val_cache);
        float dz = dh_gated * (prev_h[idx] - n[idx]);
        
        atomicAdd(&grad_prev_h[idx], dh_gated * z_val_cache + dh * (1.0f - mask_val));
        
        float n_val = temp_n_in[idx] + b_n[i] + r[idx] * temp_n_hid[idx];
        float tanh_n = tanhf(n_val);
        float d_n_pre_val = dn * (1.0f - tanh_n * tanh_n);
        d_n_pre[idx] = d_n_pre_val;
        
        float dr = d_n_pre_val * temp_n_hid[idx];
        
        float r_val = temp_r_in[idx] + temp_r_hid[idx] + b_r[i];
        float z_val = temp_z_in[idx] + temp_z_hid[idx] + b_z[i];
        
        float sig_r = 1.0f / (1.0f + expf(-r_val));
        float sig_z = 1.0f / (1.0f + expf(-z_val));
        
        float d_r_pre_val = dr * sig_r * (1.0f - sig_r);
        float d_z_pre_val = dz * sig_z * (1.0f - sig_z);
        
        d_r_pre[idx] = d_r_pre_val;
        d_z_pre[idx] = d_z_pre_val;
        
        if (!isnan(d_n_pre_val) && !isinf(d_n_pre_val)) atomicAdd(&b_n_grad[i], d_n_pre_val);
        if (!isnan(d_r_pre_val) && !isinf(d_r_pre_val)) atomicAdd(&b_r_grad[i], d_r_pre_val);
        if (!isnan(d_z_pre_val) && !isinf(d_z_pre_val)) atomicAdd(&b_z_grad[i], d_z_pre_val);
        
        grad_temp_n_hid[idx] = d_n_pre_val * r[idx];
    }
}

extern "C" __global__ void sgd_step_kernel(
    float* weights, const float* grads, float lr, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grads[idx];
        if (!isnan(g) && !isinf(g)) {
            weights[idx] -= lr * g;
        }
    }
}

extern "C" __global__ void clip_grad_norm_kernel_pass1(
    const float* grads, float* block_sums, int cols
) {
    extern __shared__ float sdata[];
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int idx = b * cols + tid;
    
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = grads[b * cols + i];
        if (!isnan(g) && !isinf(g)) {
            local_sum += g * g;
        }
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) block_sums[b] = sdata[0];
}

extern "C" __global__ void clip_grad_norm_kernel_pass2(
    float* grads, const float* block_sums, float max_norm, int cols
) {
    int b = blockIdx.y;
    int tid = threadIdx.x;
    float norm = sqrtf(block_sums[b]);
    
    if (norm > max_norm && norm > 0.0f) {
        float scale = max_norm / norm;
        for (int i = tid; i < cols; i += blockDim.x) {
            grads[b * cols + i] *= scale;
        }
    } else if (isnan(norm) || isinf(norm)) {
        for (int i = tid; i < cols; i += blockDim.x) {
            grads[b * cols + i] = 0.0f;
        }
    }
}

extern "C" __global__ void tanh_bias_kernel(
    const float* q_in,
    const float* bias,
    float* query,
    int rows,
    int cols
) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        int idx = r * cols + c;
        query[idx] = tanhf(q_in[idx] + bias[c]);
    }
}

extern "C" __global__ void softmax_rows_kernel(
    float* scores,
    int batch_size,
    int num_keys,
    float scale
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        int offset = b * num_keys;
        
        // Find max
        float max_val = -1e30f;
        for (int i = 0; i < num_keys; ++i) {
            float val = scores[offset + i] * scale;
            if (val > max_val) max_val = val;
        }
        
        // Sum exp
        float sum_exp = 0.0f;
        for (int i = 0; i < num_keys; ++i) {
            sum_exp += expf(scores[offset + i] * scale - max_val);
        }
        
        for (int i = 0; i < num_keys; ++i) {
            scores[offset + i] = expf(scores[offset + i] * scale - max_val) / (sum_exp + 1e-9f);
        }
    }
}

extern "C" __global__ void embedding_backward_kernel(
    const float* d_x,
    const int* inputs,
    float* d_w_emb_grad,
    int batch_size,
    int embed_dim,
    int vocab_size
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && i < embed_dim) {
        int token = inputs[b];
        if (token < vocab_size) {
            float g = d_x[b * embed_dim + i];
            if (!isnan(g) && !isinf(g)) {
                atomicAdd(&d_w_emb_grad[token * embed_dim + i], g);
            }
        }
    }
}

extern "C" __global__ void lion_step_kernel(
    float* weights, 
    float* grads,
    float* momentum, 
    float lr, 
    float beta1, 
    float beta2, 
    float wd, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grads[idx];
        if (isnan(g) || isinf(g)) {
            grads[idx] = 0.0f; 
            return;
        }

        float m = momentum[idx];
        
        // Evolved sign update
        float update = beta1 * m + (1.0f - beta1) * g;
        float sign_update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);
        
        float eff_grad = sign_update + wd * weights[idx];
        weights[idx] -= lr * eff_grad;
        grads[idx] = eff_grad;
        
        momentum[idx] = beta2 * m + (1.0f - beta2) * g;
    }
}

extern "C" __global__ void rmsnorm_forward_kernel(
    const float* x,
    const float* weight,
    float* y,
    float* rms_inv_out,
    int rows,
    int cols,
    float eps
) {
    int r = blockIdx.y;
    int tid = threadIdx.x;
    if (r >= rows) return;
    
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = x[r * cols + c];
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float rms_inv = 1.0f / sqrtf(sdata[0] / cols + eps);
    if (tid == 0 && rms_inv_out) {
        rms_inv_out[r] = rms_inv;
    }
    
    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = r * cols + c;
        y[idx] = x[idx] * rms_inv * weight[c];
    }
}

extern "C" __global__ void rmsnorm_backward_kernel(
    const float* x,
    const float* weight,
    const float* dy,
    const float* rms_inv,
    float* dx,
    float* d_weight_accum,
    int rows,
    int cols
) {
    int r = blockIdx.y;
    int tid = threadIdx.x;
    if (r >= rows) return;
    
    float r_inv = rms_inv[r];
    
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = r * cols + c;
        local_sum += dy[idx] * weight[c] * x[idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_dy_x = sdata[0];
    __syncthreads();
    
    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = r * cols + c;
        float dy_val = dy[idx];
        float x_val = x[idx];
        
        float dx_val = r_inv * (dy_val * weight[c] - x_val * r_inv * r_inv * sum_dy_x / cols);
        dx[idx] = dx_val;
        
        float dw = dy_val * x_val * r_inv;
        if (!isnan(dw) && !isinf(dw)) {
            atomicAdd(&d_weight_accum[c], dw);
        }
    }
}

extern "C" __global__ void mingru_forward_kernel(
    const float* temp_z_in, const float* b_z,
    const float* temp_h_in, const float* b_h,
    const float* prev_h,
    float* z_out, float* h_tilde_out, float* h_out,
    int rows, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < rows && i < cols) {
        int idx = b * cols + i;
        float z_val = temp_z_in[idx] + b_z[i];
        float h_tilde_val = temp_h_in[idx] + b_h[i];
        
        float z = 1.0f / (1.0f + expf(-z_val));
        z_out[idx] = z;
        h_tilde_out[idx] = h_tilde_val;
        
        float prev_h_val = prev_h[idx];
        h_out[idx] = (1.0f - z) * prev_h_val + z * h_tilde_val;
    }
}

extern "C" __global__ void mingru_backward_kernel(
    const float* grad_h,
    const float* z,
    const float* prev_h,
    const float* h_tilde,
    float* grad_prev_h,
    float* d_z_pre,
    float* d_h_pre,
    float* b_z_grad,
    float* b_h_grad,
    int rows, int cols
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < rows && i < cols) {
        int idx = b * cols + i;
        float dh = grad_h[idx];
        float z_val = z[idx];
        float prev_h_val = prev_h[idx];
        float h_tilde_val = h_tilde[idx];
        
        float dz = dh * (h_tilde_val - prev_h_val);
        float dz_pre_val = dz * z_val * (1.0f - z_val);
        d_z_pre[idx] = dz_pre_val;
        
        float dh_pre_val = dh * z_val;
        d_h_pre[idx] = dh_pre_val;
        
        grad_prev_h[idx] = dh * (1.0f - z_val);
        
        if (!isnan(dz_pre_val) && !isinf(dz_pre_val)) {
            atomicAdd(&b_z_grad[i], dz_pre_val);
        }
        if (!isnan(dh_pre_val) && !isinf(dh_pre_val)) {
            atomicAdd(&b_h_grad[i], dh_pre_val);
        }
    }
}

extern "C" __global__ void deltanet2_forward_kernel(
    const float* temp_k_in,
    const float* temp_v_in,
    const float* temp_q_in,
    const float* temp_b_in,
    const float* temp_w_in,
    const float* temp_alpha_in,
    const float* S_prev,
    float* S_next,
    float* y_out,
    float* k_out,
    float* v_out,
    float* q_out,
    float* b_out,
    float* w_out,
    float* alpha_out,
    int batch_size,
    int d
) {
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || j >= d) return;
    
    int step_offset = b * d;
    int s_offset = b * d * d;
    
    float k_val = temp_k_in[step_offset + j];
    float v_val = temp_v_in[step_offset + j];
    float q_val = temp_q_in[step_offset + j];
    
    float b_gate = 1.0f / (1.0f + expf(-temp_b_in[step_offset + j]));
    float w_gate = 1.0f / (1.0f + expf(-temp_w_in[step_offset + j]));
    float a_gate = 1.0f / (1.0f + expf(-temp_alpha_in[step_offset + j]));
    
    k_out[step_offset + j] = k_val;
    v_out[step_offset + j] = v_val;
    q_out[step_offset + j] = q_val;
    b_out[step_offset + j] = b_gate;
    w_out[step_offset + j] = w_gate;
    alpha_out[step_offset + j] = a_gate;
    
    __syncthreads();
    
    float row_val = 0.0f;
    for (int m = 0; m < d; ++m) {
        row_val += b_out[step_offset + m] * k_out[step_offset + m] * alpha_out[step_offset + m] * S_prev[s_offset + m * d + j];
    }
    
    float w_j = w_out[step_offset + j];
    float v_j = v_out[step_offset + j];
    
    for (int i = 0; i < d; ++i) {
        float a_i = alpha_out[step_offset + i];
        float k_i = k_out[step_offset + i];
        float s_prev_val = S_prev[s_offset + i * d + j];
        
        float s_next_val = a_i * s_prev_val - k_i * row_val + k_i * w_j * v_j;
        
        // Smooth differentiable soft-saturation (Differentiable Softsign)
        float C = 5.0f;
        float s_ratio = s_next_val / C;
        s_next_val = s_next_val / sqrtf(1.0f + s_ratio * s_ratio);
        
        S_next[s_offset + i * d + j] = s_next_val;
    }
    
    __syncthreads();
    
    for (int i = 0; i < d; ++i) {
        float val = S_next[s_offset + i * d + j] * q_val;
        atomicAdd(&y_out[step_offset + i], val);
    }
}

extern "C" __global__ void deltanet2_backward_pass1_kernel(
    const float* dy,
    const float* S_prev,
    const float* k,
    const float* v,
    const float* q,
    const float* b_gate,
    const float* w_gate,
    const float* alpha_gate,
    const float* d_S,
    float* dq,
    float* dv,
    float* dw_pre,
    float* G_temp,
    float* R_temp,
    int batch_size,
    int d
) {
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || j >= d) return;
    
    int step_offset = b * d;
    int s_offset = b * d * d;
    
    float q_j = q[step_offset + j];
    float w_j = w_gate[step_offset + j];
    float v_j = v[step_offset + j];
    
    float row_val = 0.0f;
    for (int m = 0; m < d; ++m) {
        row_val += b_gate[step_offset + m] * k[step_offset + m] * alpha_gate[step_offset + m] * S_prev[s_offset + m * d + j];
    }
    
    R_temp[step_offset + j] = w_j * v_j - row_val;
    
    float dq_sum = 0.0f;
    float G_j = 0.0f;
    for (int i = 0; i < d; ++i) {
        float s_t_val = alpha_gate[step_offset + i] * S_prev[s_offset + i * d + j] - k[step_offset + i] * row_val + k[step_offset + i] * w_j * v_j;
        float dy_val = dy[step_offset + i];
        dq_sum += s_t_val * dy_val;
        
        float dS_total_val = d_S[s_offset + i * d + j] + dy_val * q_j;
        G_j += dS_total_val * k[step_offset + i];
    }
    
    dq[step_offset + j] = dq_sum;
    G_temp[step_offset + j] = G_j;
    dv[step_offset + j] = w_j * G_j;
    dw_pre[step_offset + j] = G_j * v_j * w_j * (1.0f - w_j);
}

extern "C" __global__ void deltanet2_backward_pass2_kernel(
    const float* dy,
    const float* S_prev,
    const float* k,
    const float* q,
    const float* b_gate,
    const float* alpha_gate,
    const float* d_S,
    const float* G_temp,
    const float* R_temp,
    float* d_S_prev,
    float* dk,
    float* db_pre,
    float* dalpha_pre,
    int batch_size,
    int d
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || i >= d) return;
    
    int step_offset = b * d;
    int s_offset = b * d * d;
    
    float dy_i = dy[step_offset + i];
    float k_i = k[step_offset + i];
    float b_i = b_gate[step_offset + i];
    float a_i = alpha_gate[step_offset + i];
    
    float H_i = 0.0f;
    for (int j = 0; j < d; ++j) {
        H_i += S_prev[s_offset + i * d + j] * G_temp[step_offset + j];
    }
    
    db_pre[step_offset + i] = -k_i * a_i * H_i * b_i * (1.0f - b_i);
    
    float dalpha_sum = 0.0f;
    float dk_sum = 0.0f;
    for (int j = 0; j < d; ++j) {
        float q_j = q[step_offset + j];
        float dS_total_val = d_S[s_offset + i * d + j] + dy_i * q_j;
        float G_j = G_temp[step_offset + j];
        
        float diff = dS_total_val - b_i * k_i * G_j;
        d_S_prev[s_offset + i * d + j] = a_i * diff;
        
        dalpha_sum += S_prev[s_offset + i * d + j] * diff;
        dk_sum += dS_total_val * R_temp[step_offset + j];
    }
    
    dalpha_pre[step_offset + i] = dalpha_sum * a_i * (1.0f - a_i);
    dk[step_offset + i] = dk_sum - b_i * a_i * H_i;
}

extern "C" __global__ void compute_gate_energy_kernel(
    const float* pre_activation,
    float* energies, // size: batch_size * seq_len
    int rows,        // batch_size * seq_len
    int cols         // hidden_size
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        int offset = r * cols;
        for (int c = 0; c < cols; ++c) {
            float val = pre_activation[offset + c];
            if (val > 0.0f) {
                sum += val;
            }
        }
        energies[r] = sum / (float)cols;
    }
}


