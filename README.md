# Interactive Spin Model Simulator

A real-time, interactive simulator for two classical spin models from statistical mechanics — the **2D Ising model** and the **XY model** — rendered live at 800×800 pixels using SDL2. The simulation runs on CPU threads via OpenMP, with an optional GPU backend compiled as a CUDA DLL for massively parallel execution.

---


## Screenshots / Demo

> **TODO:** 


<img width="800" height="450" alt="VideoProject2-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/d8a6e84b-8eee-4586-8cdd-15600579f198" />





---

## The Ising Model

### The Hamiltonian

The 2D Ising model places a spin $s_i \in \{-1, +1\}$ on every site $i$ of a square lattice. The total energy of a configuration is given by the Hamiltonian:

$$H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i$$

where:

- $\langle i,j \rangle$ denotes a sum over **nearest-neighbour pairs** (up, down, left, right),
- $J$ is the coupling constant (set to 1 here — ferromagnetic),
- $h$ is an external magnetic field bias.

### Energy Change on a Single Flip

When a single spin at site $(x, y)$ is flipped ($s \to -s$), the change in energy involves only its four neighbours:

$$\Delta E = 2 s_i \left( \sum_{\delta \in \text{neighbour(i)}} s_{i+\delta} + h \right)$$

The factor of 2 arises because both the old and new energies involve the same neighbours, with opposite sign. This local formula is what makes Metropolis sampling tractable — you never have to recompute the full lattice energy.

### Phase Transition

The Ising model on a 2D square lattice has an exact analytical solution (Onsager, 1944). The critical temperature is:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269$$

(in units where $J = k_B = 1$). Below $T_c$, the system spontaneously magnetises — spins align into large domains. Above $T_c$, thermal fluctuations destroy long-range order and the lattice appears disordered. At $T_c$ itself, the domain boundaries become fractal and the correlation length diverges.

In this simulator, you can sweep through this transition in real time using the arrow keys. The default starting temperature is `T = 1.5` (ordered phase). Raising it above ~2.27 will show the disordering transition visually.

---

## The XY Model

### Continuous Spins

The XY model generalises the Ising model by replacing the discrete $\pm 1$ spin with a **continuous angle** $\theta_i \in [0, 2\pi)$, representing a unit vector pointing in the plane. The Hamiltonian is:

$$H = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j) - h \sum_i \cos(\theta_i)$$

The $\cos(\theta_i - \theta_j)$ term is minimised when neighbouring spins are aligned (parallel unit vectors). The external field $h$ biases spins toward $\theta = 0$.

### Energy Change on a Single Update

A Metropolis step proposes a small angle perturbation $\theta_i \to \theta_i + \delta$, where $\delta$ is drawn uniformly from $[-\Delta_{\max}, +\Delta_{\max}]$ with $\Delta_{\max} = \pi/4$. The local energy change is:

$$\Delta E = -\sum_{\delta \in \text{nn}} \left[\cos(\theta_i' - \theta_{i+\delta}) - \cos(\theta_i - \theta_{i+\delta})\right] - h\left[\cos(\theta_i') - \cos(\theta_i)\right]$$

### Visualisation

Each spin angle $\theta_i \in [0, 2\pi)$ maps directly to a **hue** in HSV colour space (hue = $\theta / 2\pi$, saturation = 1, value = 1). This means:

- Regions of uniform colour → aligned spin domains,
- Smooth colour gradients → slowly varying spin textures,
- Colour singularities (a point where all hues meet) → **topological vortices**, which are the hallmark of the Berezinskii–Kosterlitz–Thouless (BKT) transition.

The BKT transition (around $T_{BKT} \approx 0.89$ for $J = 1$) is a phase transition driven by the unbinding of vortex–antivortex pairs, rather than a conventional symmetry-breaking transition. The default starting temperature is set exactly at this value.

---

## The Metropolis–Hastings Algorithm

Both models use the **Metropolis–Hastings** acceptance rule. For a proposed move with energy change $\Delta E$:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta E \leq 0 \\ e^{-\Delta E / T} & \text{if } \Delta E > 0 \end{cases}$$

This satisfies **detailed balance** — the condition that ensures the simulation samples from the correct Boltzmann equilibrium distribution $P \propto e^{-H/T}$ in the long-time limit. Moves that lower energy are always accepted; moves that raise energy are accepted with a Boltzmann probability that shrinks as the temperature decreases.

Random numbers are generated using a **PCG (Permuted Congruential Generator)**, a high-quality, statistically robust RNG that is both fast and thread-safe when each thread holds its own state. On the GPU, `cuRAND` states serve the same purpose, with one RNG state per lattice site.

---

## Parallelism: The Checkerboard Decomposition

A naive parallel Metropolis update has a race condition: two neighbouring threads may try to flip adjacent spins simultaneously, producing incorrect $\Delta E$ values. This project resolves that with a **red-black (checkerboard) decomposition**.

The lattice is divided into two interlocking sublattices by parity:

- **Even sites**: $(x + y) \equiv 0 \pmod{2}$  
- **Odd sites**: $(x + y) \equiv 1 \pmod{2}$

Each even site's four nearest neighbours are all odd, and vice versa. Therefore:

1. **Pass 1** — update all even sites in parallel (no two share a neighbour).
2. **Pass 2** — update all odd sites in parallel (no two share a neighbour).

This gives full data-parallelism with no locks and no data races, while still converging to the correct equilibrium. The same two-pass pattern is used identically in the OpenMP CPU code, the CPU XY model, and the CUDA GPU kernels.

---

## GPU Compilation and Plugin Architecture

### Dual-Plugin System

The GPU backend is decoupled from the main executable through a **DLL plugin system**, allowing the C application to run without a CUDA-capable GPU and to swap out physics models seamlessly. We maintain two separate plugins:
- `xy_gpu.dll`: Evaluates the continuous-spin XY model (using double-precision floats).
- `ising.dll`: Evaluates the discrete Ising model (using integer spins).

**Step 1 — Build the CUDA DLLs** (`gpu_plugin/build.bat`):

```bat
cd gpu_plugin
build.bat
```

This invokes `nvcc.exe` to compile both `xy_gpu.cu` and `ising_gpu.cu`. `nvcc` separates device code from host code and routes each to the appropriate compiler — PTX assembly for the GPU, and MSVC (`cl.exe`) for the CPU portions. The result is two `.dll` files containing both the host-side management code and the embedded GPU binaries.

**Step 2 — Build the main executables** (`build_main.bat`):

```bat
build_main.bat
```

The main binaries are compiled with GCC (MinGW/UCRT64) and have **no CUDA dependency** at compile time. They only need the `.dll` plugins to be present at runtime.

### Runtime DLL Loading

`gpu_loader.c` uses the Windows `LoadLibraryA` / `GetProcAddress` API to bind the DLL's exported functions into a `GpuPlugin` struct of function pointers at runtime:

```c
typedef struct {
    pfn_gpu_init          init;
    pfn_gpu_destroy       destroy;
    pfn_gpu_upload_grid   upload_grid;
    pfn_gpu_update_grid   update_grid;
    pfn_gpu_render_pixels render_pixels;
    bool loaded;
} GpuPlugin;
```

If the DLL is missing or any symbol fails to resolve, `gpu_plugin_load` returns `false` and the application transparently falls back to the multi-threaded CPU path. The window title itself reflects which backend is active (`"XY Model -- GPU"` vs `"XY Model -- CPU"`).

### GPU Memory Lifecycle

All simulation state lives in GPU VRAM for the duration of the run:

| Call | Action |
|---|---|
| `gpu_init` | `cudaMalloc` the grid, pixel buffer, and per-site RNG states |
| `gpu_upload_grid` | `cudaMemcpy` initial random state Host → Device |
| `gpu_update_grid` | Run two parity kernel passes on device |
| `gpu_render_pixels` | Kernel converts angles → HSV pixels, then `cudaMemcpy` Device → Host |
| `gpu_destroy` | `cudaFree` all allocations |

The pixel buffer is the only data that crosses the PCIe bus every frame. The spin grid itself stays on the GPU, eliminating the most expensive transfer.

---

## Hysteresis Sweeps & Python Plotter

Alongside the interactive visualisers, this project includes highly-optimised, non-interactive "sweepers" that calculate the Net Magnetisation ($M$) across a sweeping external magnetic field ($h$) to generate magnetic hysteresis loops.

These sweepers utilise the GPU to calculate parallel reductions (via CUDA `atomicAdd`) to find the total sum of the grid's magnetisation extremely quickly. They output the results as `.csv` files.

You can instantly generate beautiful `matplotlib` graphs from these CSV files by using the included python script in the `plotter/` virtual environment.

---

## Execution Scripts

To make running and building everything frictionless, a `Scripts/` directory is provided. You can double-click any of these `.bat` files from Windows File Explorer, or run them from the terminal:

| Script | Purpose |
|---|---|
| `build_all.bat` | Compiles the `xy_gpu.dll` and `ising.dll` CUDA plugins and completely rebuilds all the `C` executables. |
| `run_ising_interactive.bat` | Launches the classic `Ising.exe` simulation. |
| `run_xy_interactive.bat` | Launches the `xy_interactive.exe` (XY Model GPU). |
| `run_ising_cpu_hysteresis.bat` | Uses the CPU to run the Ising model and generate `ising_hysteresis.csv`. |
| `run_ising_gpu_hysteresis.bat` | Uses the GPU to run the pure integer Ising model and generate `ising_gpu_hysteresis.csv`. |
| `run_xy_gpu_hysteresis.bat` | Uses the GPU to run the XY model and generate `xy_hysteresis.csv`. |
| `run_plotter.bat` | Feeds the generated `.csv` files into the Python script to produce graphical plots. |

---

## Building Manually

### Prerequisites

| Tool | Purpose |
|---|---|
| GCC via MSYS2 (UCRT64) | Compile the main C executable |
| SDL2 + SDL2_ttf | Windowing, rendering, fonts |
| NVIDIA CUDA Toolkit v12.x | Compile the GPU plugin DLL |
| Visual Studio Build Tools (MSVC) | Required by `nvcc` for host compilation |

### Build the GPU Plugins (requires CUDA)

```bat
cd gpu_plugin
build.bat
```

This produces `xy_gpu.dll` and `ising.dll` in the project root.

### Build the Main Executables

```bat
build_main.bat
```

This produces `xy_interactive.exe` and `Ising.exe`. 

> If you do not have a CUDA-capable GPU, skip the first step. The application will detect the missing DLLs at startup and fall back to OpenMP automatically.

---

## Controls

| Key | Action |
|---|---|
| `↑` / `↓` | Increase / decrease temperature $T$ |
| `←` / `→` | Decrease / increase external field $h$ |
| Close window | Quit |

The current values of $T$ and $h$ are displayed as a HUD overlay in the top-left corner of the window.


---

## Dependencies

- [SDL2](https://www.libsdl.org/) — cross-platform window and rendering
- [SDL2_ttf](https://github.com/libsdl-org/SDL_ttf) — TrueType font rendering for the HUD
- [OpenMP](https://www.openmp.org/) — CPU thread parallelism
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — GPU kernel compilation
- [cuRAND](https://docs.nvidia.com/cuda/curand/) — per-thread GPU random number generation
- [PCG Random](https://www.pcg-random.org/) — fast, high-quality CPU RNG (bundled in `utils/`)
