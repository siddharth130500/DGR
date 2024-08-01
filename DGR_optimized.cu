#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void computeOmega(float* d_pc, float* d_omega, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float x = d_pc[4 * idx];
        float y = d_pc[4 * idx + 1];
        float z = d_pc[4 * idx + 2];
        d_omega[idx] = asinf(z / sqrtf(x * x + y * y + z * z)) * 180.0f / 3.141592653589793f;
    }
}

__global__ void linspace(float* d_Gv, float start, float end, int num_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        d_Gv[idx] = start + idx * (end - start) / (num_values - 1);
    }
}

__global__ void createClusters(
    float* d_pc, float* d_omega, float* d_Gv, float* d_Pi,
    int num_points, int Nv, int* d_Pi_counts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float omega = d_omega[idx];

        int cluster_idx = -1;
        float min_diff = FLT_MAX;
        for (int i = 0; i < Nv + 1; i++) {
            float diff = fabsf(d_Gv[i] - omega);
            if (diff < min_diff) {
                min_diff = diff;
                cluster_idx = i;
            }
        }

        int count = atomicAdd(&d_Pi_counts[cluster_idx], 1);
        int start_idx = cluster_idx * num_points;

        d_Pi[(start_idx + count) * 4] = d_pc[idx * 4];
        d_Pi[(start_idx + count) * 4 + 1] = d_pc[idx * 4 + 1];
        d_Pi[(start_idx + count) * 4 + 2] = d_pc[idx * 4 + 2];
        d_Pi[(start_idx + count) * 4 + 3] = d_pc[idx * 4 + 3];
    }
}

__global__ void resample(
    float* d_Pi, float* d_Pi_star, int* d_angle_set,
    int* d_Pi_counts, int Nv, int Ndv, int Ndh,
    float* d_Gdv, float* d_Gdh, int num_points, float Tnorm, int w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int cluster_idx = 0; cluster_idx < Nv + 1; cluster_idx++) {
      if (idx < d_Pi_counts[cluster_idx]) {
          int point_idx = cluster_idx * num_points + idx;

          float pa[4] = {
              d_Pi[point_idx * 4],
              d_Pi[point_idx * 4 + 1],
              d_Pi[point_idx * 4 + 2],
              d_Pi[point_idx * 4 + 3]
          };

          // Initialize ptsw to contain points from neighboring clusters
          float ptsw[10000][4];
          int ptsw_count = 0;

          for (int k = -w / 2; k <= w / 2; k++) {
              if (cluster_idx + k >= 0 && cluster_idx + k < Nv && k != 0) {
                  for (int i = 0; i < d_Pi_counts[cluster_idx + k]; i++) {
                      int neighbor_idx = (cluster_idx + k) * num_points + i;
                      ptsw[ptsw_count][0] = d_Pi[neighbor_idx * 4];
                      ptsw[ptsw_count][1] = d_Pi[neighbor_idx * 4 + 1];
                      ptsw[ptsw_count][2] = d_Pi[neighbor_idx * 4 + 2];
                      ptsw[ptsw_count][3] = d_Pi[neighbor_idx * 4 + 3];
                      ptsw_count++;
                  }
              }
          }

          // Find neighbor points ptb in ptsw
          float norm_mu = 0.0f;
          int count = 0;

          for (int i = 0; i < ptsw_count; i++) {
              float distance = sqrtf(
                  (ptsw[i][0] - pa[0]) * (ptsw[i][0] - pa[0]) +
                  (ptsw[i][1] - pa[1]) * (ptsw[i][1] - pa[1]) +
                  (ptsw[i][2] - pa[2]) * (ptsw[i][2] - pa[2])
              );
              if (distance < Tnorm) {
                  norm_mu += distance;
                  count++;
              }
          }

          if (count > 0) {
              norm_mu /= count;
          }

          // Compute output point pout
          float omega_out = asinf(pa[2] / sqrtf(pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2]));
          float phi_out = atan2f(pa[1], pa[0]);

          int omega_idx = -1;
          float min_diff = FLT_MAX;
          for (int i = 0; i < Ndv + 1; i++) {
              float diff = fabsf(d_Gdv[i] - omega_out);
              if (diff < min_diff) {
                  min_diff = diff;
                  omega_idx = i;
              }
          }
          omega_out = d_Gdv[omega_idx];

          int phi_idx = -1;
          min_diff = FLT_MAX;
          for (int i = 0; i < Ndh + 1; i++) {
              float diff = fabsf(d_Gdh[i] - phi_out);
              if (diff < min_diff) {
                  min_diff = diff;
                  phi_idx = i;
              }
          }
          phi_out = d_Gdh[phi_idx];

          int angle_set_idx = omega_idx * Ndh + phi_idx;
          float r = 0.0f;
          if (atomicCAS(&d_angle_set[angle_set_idx], 0, 1) == 0 && norm_mu > 0) {
              r = norm_mu;
          }
          else {
              r = pa[0] * pa[0] +
                  pa[1] * pa[1] +
                  pa[2] * pa[2];
              r = sqrtf(r);
          }

          pa[0] = r * cosf(omega_out) * cosf(phi_out);
          pa[1] = r * cosf(omega_out) * sinf(phi_out);
          pa[2] = r * sinf(omega_out);

          int out_idx = atomicAdd(&d_Pi_counts[Nv+1], 1);
          d_Pi_star[out_idx * 4] = pa[0];
          d_Pi_star[out_idx * 4 + 1] = pa[1];
          d_Pi_star[out_idx * 4 + 2] = pa[2];
          d_Pi_star[out_idx * 4 + 3] = pa[3];
      }
    }
}


void read_lidar(const std::string& path, thrust::host_vector<float>& h_pc) {
    // Implement file reading into h_pc
    // Open the binary file
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }

    // Determine the size of the file
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(float) != 0) {
        std::cerr << "File size is not a multiple of the size of float." << std::endl;
        return;
    }

    // Calculate the number of floats in the file
    std::size_t num_floats = size / sizeof(float);

    // Resize the thrust::host_vector to accommodate the data
    h_pc.resize(num_floats);

    // Read the data from the file into the thrust::host_vector
    if (!file.read(reinterpret_cast<char*>(h_pc.data()), size)) {
        std::cerr << "Error reading file data." << std::endl;
    }

    // Close the file
    file.close();
}

int main() {
    std::string path = "/content/000005.bin";
    thrust::host_vector<float> h_pc;
    read_lidar(path, h_pc);

    int num_points = h_pc.size() / 4;
    int Nv = 64;
    int Ndv = 40;
    int Ndh = 720;
    int w = 3;
    float Tnorm = 0.25;
    thrust::device_vector<float> d_pc = h_pc;
    thrust::device_vector<float> d_omega(num_points);
    thrust::device_vector<float> d_Gv(Nv + 1);
    thrust::device_vector<float> d_Gdv(Ndv + 1);
    thrust::device_vector<float> d_Gdh(Ndh + 1);
    thrust::device_vector<float> d_Pi((Nv + 1) * 4 * num_points);
    thrust::device_vector<float> d_Pi_star(num_points * 4);
    thrust::device_vector<int> d_angle_set(Nv * Ndh, 0);
    thrust::device_vector<int> d_Pi_counts(Nv + 1 + 1, 0);  // +1 for output count


    linspace<<<(Nv + 1 + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_Gv.data()), -24.8f, 2.0f, Nv + 1);
    linspace<<<(Ndv + 1 + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_Gdv.data()), -24.8f, 2.0f, Ndv + 1);
    linspace<<<(Ndh + 1 + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_Gdh.data()), -180.0f, 180.0f, Ndh + 1);

    computeOmega<<<(num_points + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_pc.data()), thrust::raw_pointer_cast(d_omega.data()), num_points);
    createClusters<<<(num_points + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_pc.data()), thrust::raw_pointer_cast(d_omega.data()), thrust::raw_pointer_cast(d_Gv.data()), thrust::raw_pointer_cast(d_Pi.data()), num_points, Nv, thrust::raw_pointer_cast(d_Pi_counts.data()));

    resample<<<(num_points + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_Pi.data()), thrust::raw_pointer_cast(d_Pi_star.data()),
        thrust::raw_pointer_cast(d_angle_set.data()),
        thrust::raw_pointer_cast(d_Pi_counts.data()), Nv, Ndv, Ndh,
        thrust::raw_pointer_cast(d_Gdv.data()), thrust::raw_pointer_cast(d_Gdh.data()),
        num_points, Tnorm, w
    );
    cudaDeviceSynchronize();


    return 0;
}
