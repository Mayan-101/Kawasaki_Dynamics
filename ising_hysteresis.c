#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "utils/pcg_random.h"

#define WIDTH 800
#define HEIGHT 800

#define IDX(x, y) ((y) * WIDTH + (x))

static inline int wrap(int i, int max)
{
    return (i + max) % max;
}

void update_grid(int *grid, int height, int width, double T, double h)
{
#pragma omp parallel
    {
        uint64_t seed = (uint64_t)time(NULL) ^ (omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
        pcg_seed(seed, 42);
#pragma omp for
        for (int idx = 0; idx < height * width; idx++)
        {
            int x = idx % width;
            int y = idx / width;
            if (((x + y) & 1) != 0)
                continue;

            int current = grid[IDX(x, y)];
            int neighbor_sum = grid[IDX(wrap(x - 1, width), y)] + grid[IDX(wrap(x + 1, width), y)] + grid[IDX(x, wrap(y - 1, height))] + grid[IDX(x, wrap(y + 1, height))];

            double delta_E = 2.0 * current * (neighbor_sum + h);

            if (delta_E <= 0.0 || (T > 0.0 && pcg_rand_double() < exp(-delta_E / T)))
            {
                grid[IDX(x, y)] = -current;
            }
        }
    }

#pragma omp parallel
    {
        uint64_t seed = (uint64_t)time(NULL) ^ (omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
        pcg_seed(seed, 42);
#pragma omp for
        for (int idx = 0; idx < height * width; idx++)
        {
            int x = idx % width;
            int y = idx / width;
            if (((x + y) & 1) != 1)
                continue;

            int current = grid[IDX(x, y)];
            int neighbor_sum = grid[IDX(wrap(x - 1, width), y)] + grid[IDX(wrap(x + 1, width), y)] + grid[IDX(x, wrap(y - 1, height))] + grid[IDX(x, wrap(y + 1, height))];

            double delta_E = 2.0 * current * (neighbor_sum + h);

            if (delta_E <= 0.0 || (T > 0.0 && pcg_rand_double() < exp(-delta_E / T)))
            {
                grid[IDX(x, y)] = -current;
            }
        }
    }
}

static double measure_M(const int *grid)
{
    long long sum = 0;
    for (int i = 0; i < HEIGHT * WIDTH; i++)
        sum += grid[i];
    return (double)sum / (double)(HEIGHT * WIDTH);
}

static void init_all_up(int *grid)
{
    for (int i = 0; i < HEIGHT * WIDTH; i++)
        grid[i] = 1;
}

void run_hysteresis(double T, double h_max, int h_steps,
                    int equil_sweeps, int meas_sweeps)
{
    int *grid = (int *)malloc(HEIGHT * WIDTH * sizeof(int));
    if (!grid) { fprintf(stderr, "Memory allocation failed\n"); return; }

    FILE *fp = fopen("ising_hysteresis.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open ising_hysteresis.csv\n"); free(grid); return; }

    fprintf(fp, "h,M,branch\n");

    printf("Hysteresis sweep  T=%.4f  h_max=%.4f  h_steps=%d"
           "  equil=%d  meas=%d\n", T, h_max, h_steps, equil_sweeps, meas_sweeps);
    printf("Output -> ising_hysteresis.csv\n");

    init_all_up(grid);
    printf("Branch 0 (descending): ");
    fflush(stdout);

    for (int step = 0; step <= h_steps; step++)
    {
        double h = h_max - (2.0 * h_max * step) / h_steps;

        for (int s = 0; s < equil_sweeps; s++)
            update_grid(grid, HEIGHT, WIDTH, T, h);

        double M_acc = 0.0;
        for (int s = 0; s < meas_sweeps; s++)
        {
            update_grid(grid, HEIGHT, WIDTH, T, h);
            M_acc += measure_M(grid);
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
            update_grid(grid, HEIGHT, WIDTH, T, h);

        double M_acc = 0.0;
        for (int s = 0; s < meas_sweeps; s++)
        {
            update_grid(grid, HEIGHT, WIDTH, T, h);
            M_acc += measure_M(grid);
        }

        fprintf(fp, "%.8f,%.8f,1\n", h, M_acc / meas_sweeps);

        if (step % (h_steps / 10) == 0)
        { printf("%.0f%% ", 100.0 * step / h_steps); fflush(stdout); }
    }
    printf("done\n");

    fclose(fp);
    free(grid);
    printf("Saved ising_hysteresis.csv\n");
}

int main(int argc, char *argv[])
{
#pragma omp parallel
    {
#pragma omp single
        printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    }

    double T = 1.5;
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

    run_hysteresis(T, h_max, h_steps, equil, meas);

    return 0;
}
