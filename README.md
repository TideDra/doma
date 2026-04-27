# Doma

**Doma** (Dog in the Manger) is a GPU utilization controller that guarantees a minimum GPU utilization percentage on NVIDIA and AMD GPUs. It dynamically adjusts GPU workload to maintain the desired utilization level, making it useful for GPU reservation, testing, and preventing GPUs from being idle.

## Features

- 🎯 **Guaranteed Minimum Utilization**: Ensures your GPUs maintain at least the specified utilization percentage.
- 🔧 **Multi-GPU Support**: Control multiple GPUs simultaneously with individual thread management.
- 🔄 **Dynamic Adjustment**: Automatically adjusts workload based on other processes using the GPU. No performance impact on your real job.
- ⚡ **Extremely Light-weight**: No PyTorch, CUDA Toolkit, or ROCm compiler is needed at runtime. NVIDIA uses the driver/NVML stack, and AMD uses embedded HSACO kernels plus ROCm HIP/AMD SMI runtime libraries.


## Installation

### From Source

```bash
# Make sure rust environment is installed
git clone https://github.com/TideDra/doma.git
cd doma
cargo build --release
```

The compiled binary will be available at `target/release/doma`.

Runtime requirements:

- NVIDIA: NVIDIA driver libraries (`libcuda.so.1`) and NVML.
- AMD: ROCm HIP runtime (`libamdhip64.so`) and AMD SMI (`libamd_smi.so`). `hipcc` and HIPRTC are not needed to run doma.

### Use compiled binary
Compiled binary can be found at [Releases](https://github.com/TideDra/doma/releases).
```bash
# Download the binary
wget https://github.com/TideDra/doma/releases/download/v0.1.0/doma-v0.1.0-x86_64-unknown-linux-gnu.tar.gz
# Unzip the binary
tar -xzf doma-v0.1.0-x86_64-unknown-linux-gnu.tar.gz
# Run the binary
./doma 50
```

Doma automatically selects an available backend. To force a specific backend:

```bash
./doma 50 --backend nvidia
./doma 50 --backend amd
```

## Usage

### Basic Usage

Set minimum GPU utilization to 50% on all available GPUs:

```bash
./doma 50
```

### Control Specific GPUs

Set minimum utilization to 70% on GPUs 0 and 2:

```bash
./doma 70 --device-ids 0,2
# or use short form
./doma 70 -d 0,2
```

For AMD GPUs, device IDs are HIP device IDs, which normally match `amd-smi list` ordering.

### Stop the Controller

Press `Ctrl+C` to gracefully stop all GPU controllers.

## How It Works

Doma operates by:

1. **Calibration**: On startup, it calibrates an embedded CUDA PTX or AMD HSACO kernel to determine how many loop iterations are needed per millisecond on your GPU
2. **Monitoring**: Uses NVML on NVIDIA and AMD SMI on AMD to monitor current GPU utilization and process activity
3. **Dynamic Adjustment**: Calculates how much additional utilization is needed to reach the target
4. **Workload Injection**: Executes calibrated CUDA/AMD kernels in time windows (100ms by default) to maintain the desired utilization
5. **Sleep Cycles**: Sleeps for the remaining time in each window to avoid exceeding the target

The controller continuously adjusts its workload based on other processes using the GPU, ensuring the total utilization stays at or above the minimum threshold. If your GPU job already reaches the threshold, **doma will not occupy the GPU.**

The AMD kernel is precompiled and embedded in the binary for common ROCm targets (`gfx90a`, `gfx942`, `gfx950`, `gfx1030`, `gfx1100`, `gfx1101`, and `gfx1102`). If you modify `src/hip/kernel.hip`, regenerate `src/hip/kernel.hsaco` with `hipcc --genco` before building a release.

## Command-Line Options

```
Usage: doma [OPTIONS] <MIN_UTIL>

Arguments:
  <MIN_UTIL>  Minimum utilization percentage (1-100)

Options:
  -d, --device-ids <DEVICE_IDS>  GPU IDs to control, separated by commas
  -m, --mem-reserve <MEM_RESERVE> GPU memory to reserve in GB when no other compute process is present
      --backend <BACKEND>         GPU backend to use [default: auto] [possible values: auto, nvidia, amd]
  -h, --help                      Print help
  -V, --version                   Print version
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**TideDra** - [gearyzhang@outlook.com](mailto:gearyzhang@outlook.com)

## Repository

[https://github.com/TideDra/doma](https://github.com/TideDra/doma)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
