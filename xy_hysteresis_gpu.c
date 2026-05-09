#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "utils/pcg_random.h"
#include "gpu_loader.h"

#define M_PI 3.14159265358979323846264338327950288
#define WIDTH 800
#define HEIGHT 800

#define TWO_PI (2.0 * M_PI)
#define DELTA_MAX (M_PI / 4.0)

#define IDX(x, y) ((y) * WIDTH + (x))

static inline int wrap(int i, int max) { return (i + max) % max; }

static inline double wrap_angle(double a)
{
    a = fmod(a, TWO_PI);
    return (a < 0.0) ? a + TWO_PI : a;
}

void update_grid(double *grid, int height, int width, double T, double h)
{
    for (int parity = 0; parity <= 1; parity++)
    {
#pragma omp parallel
        {
            uint64_t seed = (uint64_t)time(NULL) ^ ((uint64_t)omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
            pcg_seed(seed, 42);

#pragma omp for collapse(2) schedule(static)
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (((x + y) & 1) != parity)
                        continue;

                    double theta = grid[IDX(x, y)];

                    double tL = grid[IDX(wrap(x - 1, width), y)];
                    double tR = grid[IDX(wrap(x + 1, width), y)];
                    double tU = grid[IDX(x, wrap(y - 1, height))];
                    double tD = grid[IDX(x, wrap(y + 1, height))];

                    double delta = (pcg_rand_double() * 2.0 - 1.0) * DELTA_MAX;
                    double theta_new = wrap_angle(theta + delta);

                    double dE = -((cos(theta_new - tL) - cos(theta - tL)) + (cos(theta_new - tR) - cos(theta - tR)) + (cos(theta_new - tU) - cos(theta - tU)) + (cos(theta_new - tD) - cos(theta - tD))) - h * (cos(theta_new) - cos(theta));

                    if (dE <= 0.0 || (T > 0.0 && pcg_rand_double() < exp(-dE / T)))
                        grid[IDX(x, y)] = theta_new;
                }
            }
        }
    }
}

static double measure_M_cpu(const double *grid)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int i = 0; i < HEIGHT * WIDTH; i++)
    {
        sum_x += cos(grid[i]);
        sum_y += sin(grid[i]);
    }
    return sqrt(sum_x * sum_x + sum_y * sum_y) / (double)(HEIGHT * WIDTH);
}

void init_all_up(double *grid)
{
    for (int i = 0; i < HEIGHT * WIDTH; i++)
        grid[i] = 0.0; // All pointing in the same direction (theta = 0)
}

void run_hysteresis(double T, double h_max, int h_steps, int equil_sweeps, int meas_sweeps, GpuPlugin *gpu, GpuState *gpu_state, double *host_grid)
{
    FILE *fp = fopen("xy_hysteresis.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open xy_hysteresis.csv\n"); return; }

    fprintf(fp, "h,M,branch\n");

    printf("XY Hysteresis sweep  T=%.4f  h_max=%.4f  h_steps=%d"
           "  equil=%d  meas=%d\n", T, h_max, h_steps, equil_sweeps, meas_sweeps);
    printf("Output -> xy_hysteresis.csv\n");

    if (gpu_state) {
        init_all_up(host_grid);
        gpu->upload_grid(gpu_state, host_grid);
    } else {
        init_all_up(host_grid);
    }

    printf("Branch 0 (descending): ");
    fflush(stdout);

    for (int step = 0; step <= h_steps; step++)
    {
        double h = h_max - (2.0 * h_max * step) / h_steps;

        for (int s = 0; s < equil_sweeps; s++)
        {
            if (gpu_state) gpu->update_grid(gpu_state, T, h);
            else update_grid(host_grid, HEIGHT, WIDTH, T, h);
        }

        double M_acc = 0.0;
        for (int s = 0; s < meas_sweeps; s++)
        {
            if (gpu_state) {
                gpu->update_grid(gpu_state, T, h);
                M_acc += gpu->measure_M(gpu_state);
            } else {
                update_grid(host_grid, HEIGHT, WIDTH, T, h);
                M_acc += measure_M_cpu(host_grid);
            }
        }

        fprintf(fp, "%.8f,%.8f,0\n", h, M_acc / meas_sweeps);

        if (step % (h_steps / 10) == 0)
        { printf("%.0f%% ", 100.0 * step / h_steps); fflush(stdout); }
    }
    printf("done\n");

    printf("Branch 1 (ascending):  ");
    fflush(stdout);

    for (int step = 0; step <= h_steps; step++)
    {
        double h = -h_max + (2.0 * h_max * step) / h_steps;

        for (int s = 0; s < equil_sweeps; s++)
        {
            if (gpu_state) gpu->update_grid(gpu_state, T, h);
            else update_grid(host_grid, HEIGHT, WIDTH, T, h);
        }

        double M_acc = 0.0;
        for (int s = 0; s < meas_sweeps; s++)
        {
            if (gpu_state) {
                gpu->update_grid(gpu_state, T, h);
                M_acc += gpu->measure_M(gpu_state);
            } else {
                update_grid(host_grid, HEIGHT, WIDTH, T, h);
                M_acc += measure_M_cpu(host_grid);
            }
        }

        fprintf(fp, "%.8f,%.8f,1\n", h, M_acc / meas_sweeps);

        if (step % (h_steps / 10) == 0)
        { printf("%.0f%% ", 100.0 * step / h_steps); fflush(stdout); }
    }
    printf("done\n");

    fclose(fp);
    printf("Saved xy_hysteresis.csv\n");
}

int main(int argc, char *argv[])
{
#pragma omp parallel
    {
#pragma omp single
        printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    }

    GpuPlugin gpu = {0};
    bool use_gpu = gpu_plugin_load(&gpu, "xy_gpu.dll");

    double *host_grid = (double *)malloc(HEIGHT * WIDTH * sizeof(double));
    if (!host_grid)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    GpuState *gpu_state = NULL;
    if (use_gpu)
    {
        gpu_state = gpu.init(HEIGHT, WIDTH);
        printf("Simulation running on GPU\n");
    }
    else
    {
        printf("GPU unavailable -- running on CPU\n");
    }

    double T = 0.89;
    double h_max = 3.0;
    int h_steps = 300;
    int equil = 500;
    int meas = 200;

    if (argc >= 6)
    {
        T = atof(argv[1]);
        h_max = atof(argv[2]);
        h_steps = atoi(argv[3]);
        equil = atoi(argv[4]);
        meas = atoi(argv[5]);
    }
    else
    {
        printf("Usage: %s [T] [h_max] [h_steps] [equil] [meas]\n", argv[0]);
        printf("Using defaults: T=%.2f h_max=%.2f h_steps=%d equil=%d meas=%d\n\n",
               T, h_max, h_steps, equil, meas);
    }

    run_hysteresis(T, h_max, h_steps, equil, meas, &gpu, gpu_state, host_grid);

    if (use_gpu)
    {
        gpu.destroy(gpu_state);
        gpu_plugin_unload();
    }
    free(host_grid);

    return 0;
}
