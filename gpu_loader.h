#ifndef GPU_LOADER_H
#define GPU_LOADER_H

#include <stdint.h>
#include <stdbool.h>

// Opaque — never dereferenced outside the DLL
typedef struct GpuState GpuState;

// Mirror every exported function as a typed pointer
typedef GpuState *(*pfn_gpu_init)(int height, int width);
typedef void (*pfn_gpu_destroy)(GpuState *s);
typedef void (*pfn_gpu_upload_grid)(GpuState *s, const double *host_grid);
typedef void (*pfn_gpu_update_grid)(GpuState *s, double T, double h);
typedef void (*pfn_gpu_render_pixels)(GpuState *s, uint32_t *host_pixels);
typedef double (*pfn_gpu_measure_M)(GpuState *s);

typedef struct
{
    pfn_gpu_init init;
    pfn_gpu_destroy destroy;
    pfn_gpu_upload_grid upload_grid;
    pfn_gpu_update_grid update_grid;
    pfn_gpu_render_pixels render_pixels;
    pfn_gpu_measure_M measure_M;
    bool loaded;
} GpuPlugin;

// Returns true if DLL found and every symbol resolved.
// On failure the plugin is zeroed and the caller should fall back to CPU.
bool gpu_plugin_load(GpuPlugin *p, const char *dll_path);
void gpu_plugin_unload(void);

#endif