#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

#define CEIL(x, y) (((x) + (y) - 1) / (y))

//实现cuda算子
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_output, size_t rows, size_t cols) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr int wrapSize = 32; // Assuming a warp size of 32
  int wrap_id = threadIdx.x / wrapSize;
  int lane_id = threadIdx.x % wrapSize;
  __shared__ T shared_data[32];

  T val = (idx < rows && idx < cols) ? d_input[idx * cols + idx] : (T)0;
  #pragma unroll
  for(int offset = wrapSize >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  if(lane_id == 0) {
    shared_data[wrap_id] = val;
  }
  __syncthreads();
  if(wrap_id == 0) {
    int wrapnum = (blockDim.x / wrapSize);
    val = lane_id < wrapnum ? shared_data[lane_id] : (T)0;
    __syncthreads();
    #pragma unroll(4)
    for(int offset = wrapSize >> 1; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(lane_id == 0) {
      atomicAdd(d_output, val);
    }
  }
}
// template <typename T>
// __global__ void traceKernel(const T* d_input, T* d_output, size_t rows, size_t cols) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx < rows && idx < cols) {
//     atomicAdd(d_output, d_input[idx * cols + idx]);
//   }
// }

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  // return T(-1);
  T h_output = 0;
  T* d_input;
  T* d_output;
  cudaMalloc(&d_input, h_input.size() * sizeof(T));
  cudaMalloc(&d_output, sizeof(T));
  cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);
  //shared memory init
  traceKernel<<<CEIL(rows, 256), 256>>>(d_input, d_output, rows, cols);
  cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
  return h_output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
#include <float.h>


#define BLOCK_M 128
#define BLOCK_N 64

__device__ __forceinline__ float exp_func(float x) { return __expf(x); }

template <typename T>
__global__ void flash_attn_fwd_kernel(
    const T* __restrict__ Q,    // [B, L, NH, D]
    const T* __restrict__ K,    // [B, S, KH, D]
    const T* __restrict__ V,    // [B, S, KH, D]
    T* __restrict__ O,          // [B, L, NH, D]
    int B, int L, int S, int NH, int KH, int D,
    float scale, bool is_causal
) {
    // 线程定位
    int bx = blockIdx.x; // batch_idx
    int bh = blockIdx.y; // head_idx
    int tid = threadIdx.x;

    int gqa_ratio = NH / KH;
    int kh = bh / gqa_ratio;

    for (int m = 0; m < (L + BLOCK_M - 1) / BLOCK_M; ++m) {
        int q_idx = m * BLOCK_M + tid;
        
        // 每个线程维护的状态
        float row_m = -FLT_MAX;
        float row_l = 0.0f;
        float row_o[128]; // 假设 D=128, 实际应用中应根据 D 动态或模板化
        for(int d=0; d<D; ++d) row_o[d] = 0.0f;

        // 内部循环分块 (Key/Value 维度)
        for (int n = 0; n < (S + BLOCK_N - 1) / BLOCK_N; ++n) {
            // 1. 加载 K, V 到 Shared Memory (此处简化，实际需处理并发加载)
            // 2. 计算 S = Q * K^T * scale
            // 3. 处理 Causal Masking
            // 4. 更新 Online Softmax 统计量并更新 row_o
            
            for (int j = 0; j < BLOCK_N; ++j) {
                int target_s_idx = n * BLOCK_N + j;
                if (target_s_idx >= S) continue;
                if (is_causal && target_s_idx > q_idx) continue;

                float score = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float q_val = (float)Q[((bx * L + q_idx) * NH + bh) * D + d];
                    float k_val = (float)K[((bx * S + target_s_idx) * KH + kh) * D + d];
                    score += q_val * k_val;
                }
                score *= scale;

                // Online Softmax 更新
                float old_m = row_m;
                row_m = fmaxf(row_m, score);
                float exp_prev = exp_func(old_m - row_m);
                float exp_curr = exp_func(score - row_m);

                row_l = row_l * exp_prev + exp_curr;

                for (int d = 0; d < D; ++d) {
                    float v_val = (float)V[((bx * S + target_s_idx) * KH + kh) * D + d];
                    row_o[d] = row_o[d] * exp_prev + v_val * exp_curr;
                }
            }
        }

        if (q_idx < L) {
            for (int d = 0; d < D; ++d) {
                O[((bx * L + q_idx) * NH + bh) * D + d] = (T)(row_o[d] / row_l);
            }
        }
    }
}

// 包装函数
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_bytes = h_q.size() * sizeof(T);
    size_t k_bytes = h_k.size() * sizeof(T);
    size_t v_bytes = h_v.size() * sizeof(T);
    size_t o_bytes = h_o.size() * sizeof(T);

    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, k_bytes);
    cudaMalloc(&d_v, v_bytes);
    cudaMalloc(&d_o, o_bytes);

    cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Grid: [Batch, Query_Heads]
    // Block: [BLOCK_M] 每线程处理一行 Query
    dim3 grid(batch_size, query_heads);
    dim3 block(BLOCK_M);

    flash_attn_fwd_kernel<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal
    );

    cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
