#include "ising_gpu_loader.h"
#include <windows.h>
#include <stdio.h>
#include <string.h>

static HMODULE ising_dll = NULL;

#define BIND_ISING(p, T, name)                                               \
    (p)->name = (T)GetProcAddress(ising_dll, "gpu_" #name);                  \
    if (!(p)->name) {                                                        \
        fprintf(stderr, "ising_gpu_loader: missing symbol gpu_" #name "\n"); \
        FreeLibrary(ising_dll); ising_dll = NULL; return false;              \
    }

bool ising_gpu_plugin_load(IsingGpuPlugin *p, const char *dll_path)
{
    memset(p, 0, sizeof(*p));

    ising_dll = LoadLibraryA(dll_path);
    if (!ising_dll) {
        fprintf(stderr, "ising_gpu_loader: LoadLibrary(\"%s\") failed -- error %lu\n",
                dll_path, GetLastError());
        return false;
    }

    BIND_ISING(p, pfn_ising_gpu_init,          init)
    BIND_ISING(p, pfn_ising_gpu_destroy,       destroy)
    BIND_ISING(p, pfn_ising_gpu_upload_grid,   upload_grid)
    BIND_ISING(p, pfn_ising_gpu_update_grid,   update_grid)
    BIND_ISING(p, pfn_ising_gpu_render_pixels, render_pixels)
    BIND_ISING(p, pfn_ising_gpu_measure_M,     measure_M)

    p->loaded = true;
    printf("ising_gpu_loader: loaded \"%s\"\n", dll_path);
    return true;
}

void ising_gpu_plugin_unload(void)
{
    if (ising_dll) { FreeLibrary(ising_dll); ising_dll = NULL; }
}
