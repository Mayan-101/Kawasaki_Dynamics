#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TWO_PI (2.0 * 3.14159265358979323846)
#define DELTA_MAX (3.14159265358979323846 / 4.0)

#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t _e = (call);                                 \
        if (_e != cudaSuccess)                                   \
        {                                                        \
            fprintf(stderr, "CUDA %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1);                                             \
        }                                                        \
    } while (0)

struct GpuState
{
    int height, width;
    double *d_grid;
    uint32_t *d_pixels;
    curandState *d_rng;
};



__global__ void k_init_rng(curandState *states, int height, int width,
                           unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    curand_init(seed, (unsigned long long)(y * width + x), 0,
                &states[y * width + x]);
}

__global__ void k_update(double *grid, int height, int width,
                         double T, double h, int parity, curandState *rng)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    if (((x + y) & 1) != parity)
        return;

    int idx = y * width + x;
    curandState local = rng[idx];

    double theta = grid[idx];
    double tL = grid[y * width + (x - 1 + width) % width];
    double tR = grid[y * width + (x + 1) % width];
    double tU = grid[((y - 1 + height) % height) * width + x];
    double tD = grid[((y + 1) % height) * width + x];

    double delta = (curand_uniform_double(&local) * 2.0 - 1.0) * DELTA_MAX;
    double theta_new = fmod(theta + delta + TWO_PI * 10.0, TWO_PI);

    double dE = -((cos(theta_new - tL) - cos(theta - tL)) + (cos(theta_new - tR) - cos(theta - tR)) + (cos(theta_new - tU) - cos(theta - tU)) + (cos(theta_new - tD) - cos(theta - tD))) - h * (cos(theta_new) - cos(theta));

    if (dE <= 0.0 || (T > 0.0 && curand_uniform_double(&local) < exp(-dE / T)))
        grid[idx] = theta_new;

    rng[idx] = local;
}

__device__ void d_hsv_to_rgb(double hue, uint8_t *r, uint8_t *g, uint8_t *b)
{
    double h6 = hue * 6.0;
    int i = (int)h6 % 6;
    double f = h6 - (int)h6, q = 1.0 - f;
    switch (i)
    {
    case 0:
        *r = 255;
        *g = (uint8_t)(f * 255);
        *b = 0;
        break;
    case 1:
        *r = (uint8_t)(q * 255);
        *g = 255;
        *b = 0;
        break;
    case 2:
        *r = 0;
        *g = 255;
        *b = (uint8_t)(f * 255);
        break;
    case 3:
        *r = 0;
        *g = (uint8_t)(q * 255);
        *b = 255;
        break;
    case 4:
        *r = (uint8_t)(f * 255);
        *g = 0;
        *b = 255;
        break;
    case 5:
        *r = 255;
        *g = 0;
        *b = (uint8_t)(q * 255);
        break;
    default:
        *r = 0;
        *g = 0;
        *b = 0;
    }
}

__global__ void k_render(const double *grid, uint32_t *pixels,
                         int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    double hue = grid[y * width + x] / TWO_PI;
    uint8_t r, g, b;
    d_hsv_to_rgb(hue, &r, &g, &b);
    pixels[y * width + x] = (255u << 24) | (b << 16) | (g << 8) | r;
}



static dim3 blocks(int w, int h, dim3 t)
{
    return dim3((w + t.x - 1) / t.x, (h + t.y - 1) / t.y);
}



extern "C"
{

    __declspec(dllexport)
    GpuState *
    gpu_init(int height, int width)
    {
        GpuState *s = (GpuState *)malloc(sizeof(GpuState));
        s->height = height;
        s->width = width;
        CUDA_CHECK(cudaMalloc(&s->d_grid, height * width * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s->d_pixels, height * width * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s->d_rng, height * width * sizeof(curandState)));

        dim3 t(16, 16);
        k_init_rng<<<blocks(width, height, t), t>>>(s->d_rng, height, width, 42ULL);
        CUDA_CHECK(cudaDeviceSynchronize());
        return s;
    }

    __declspec(dllexport) void gpu_destroy(GpuState *s)
    {
        cudaFree(s->d_grid);
        cudaFree(s->d_pixels);
        cudaFree(s->d_rng);
        free(s);
    }

    __declspec(dllexport) void gpu_upload_grid(GpuState *s, const double *host_grid)
    {
        CUDA_CHECK(cudaMemcpy(s->d_grid, host_grid,
                              s->height * s->width * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    __declspec(dllexport) void gpu_update_grid(GpuState *s, double T, double h)
    {
        dim3 t(16, 16);
        dim3 b = blocks(s->width, s->height, t);
        for (int parity = 0; parity <= 1; parity++)
        {
            k_update<<<b, t>>>(s->d_grid, s->height, s->width, T, h, parity, s->d_rng);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    __declspec(dllexport) void gpu_render_pixels(GpuState *s, uint32_t *host_pixels)
    {
        dim3 t(16, 16);
        k_render<<<blocks(s->width, s->height, t), t>>>(
            s->d_grid, s->d_pixels, s->height, s->width);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_pixels, s->d_pixels,
                              s->height * s->width * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

} 