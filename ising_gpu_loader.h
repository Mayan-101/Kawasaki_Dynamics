#ifndef ISING_GPU_LOADER_H
#define ISING_GPU_LOADER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct IsingGpuState IsingGpuState;

typedef IsingGpuState *(*pfn_ising_gpu_init)(int height, int width);
typedef void (*pfn_ising_gpu_destroy)(IsingGpuState *s);
typedef void (*pfn_ising_gpu_upload_grid)(IsingGpuState *s, const int *host_grid);
typedef void (*pfn_ising_gpu_update_grid)(IsingGpuState *s, double T, double h);
typedef void (*pfn_ising_gpu_render_pixels)(IsingGpuState *s, uint32_t *host_pixels);
typedef double (*pfn_ising_gpu_measure_M)(IsingGpuState *s);

typedef struct
{
    pfn_ising_gpu_init init;
    pfn_ising_gpu_destroy destroy;
    pfn_ising_gpu_upload_grid upload_grid;
    pfn_ising_gpu_update_grid update_grid;
    pfn_ising_gpu_render_pixels render_pixels;
    pfn_ising_gpu_measure_M measure_M;
    bool loaded;
} IsingGpuPlugin;

bool ising_gpu_plugin_load(IsingGpuPlugin *p, const char *dll_path);
void ising_gpu_plugin_unload(void);

#endif
