#include "gpu_loader.h"
#include <windows.h>
#include <stdio.h>
#include <string.h>

static HMODULE dll = NULL;

// One macro per symbol keeps things DRY and gives a clear error on mismatch
#define BIND(p, T, name)                                                     \
    (p)->name = (T)GetProcAddress(dll, "gpu_" #name);                       \
    if (!(p)->name) {                                                        \
        fprintf(stderr, "gpu_loader: missing symbol gpu_" #name "\n");      \
        FreeLibrary(dll); dll = NULL; return false;                         \
    }

bool gpu_plugin_load(GpuPlugin *p, const char *dll_path)
{
    memset(p, 0, sizeof(*p));

    dll = LoadLibraryA(dll_path);
    if (!dll) {
        fprintf(stderr, "gpu_loader: LoadLibrary(\"%s\") failed — error %lu\n",
                dll_path, GetLastError());
        return false;
    }

    BIND(p, pfn_gpu_init,          init)
    BIND(p, pfn_gpu_destroy,       destroy)
    BIND(p, pfn_gpu_upload_grid,   upload_grid)
    BIND(p, pfn_gpu_update_grid,   update_grid)
    BIND(p, pfn_gpu_render_pixels, render_pixels)

    p->loaded = true;
    printf("gpu_loader: loaded \"%s\"\n", dll_path);
    return true;
}

void gpu_plugin_unload(void)
{
    if (dll) { FreeLibrary(dll); dll = NULL; }
}