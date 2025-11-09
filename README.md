# Doma

**Doma** (Dog in the Manger) is a GPU utilization controller that guarantees a minimum GPU utilization percentage on NVIDIA GPUs. It dynamically adjusts GPU workload to maintain the desired utilization level, making it useful for GPU reservation, testing, and preventing GPUs from being idle.

## Features

- 🎯 **Guaranteed Minimum Utilization**: Ensures your GPUs maintain at least the specified utilization percentage.
- 🔧 **Multi-GPU Support**: Control multiple GPUs simultaneously with individual thread management.
- 🔄 **Dynamic Adjustment**: Automatically adjusts workload based on other processes using the GPU. No performance affect on your real job.
- ⚡ **Extremly Light-weight**: No CUDA Runtime or PyTorch Needed. All you need is the **doma** binary (~2MB) and Nvidia Driver.


## Installation

### From Source

```bash
# Make sure rust environment is installed
git clone https://github.com/TideDra/doma.git
cd doma
cargo build --release
```

The compiled binary will be available at `target/release/doma`.

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

### Stop the Controller

Press `Ctrl+C` to gracefully stop all GPU controllers.

## How It Works

Doma operates by:

1. **Calibration**: On startup, it calibrates a CUDA kernel to determine how many loop iterations are needed per millisecond on your GPU
2. **Monitoring**: Uses NVML to monitor current GPU utilization from other processes
3. **Dynamic Adjustment**: Calculates how much additional utilization is needed to reach the target
4. **Workload Injection**: Executes calibrated CUDA kernels in time windows (100ms by default) to maintain the desired utilization
5. **Sleep Cycles**: Sleeps for the remaining time in each window to avoid exceeding the target

The controller continuously adjusts its workload based on other processes using the GPU, ensuring the total utilization stays at or above the minimum threshold. If your GPU job already reaches the threshold, **doma will not occupy the GPU.**

## Command-Line Options

```
Usage: doma [OPTIONS] <MIN_UTIL>

Arguments:
  <MIN_UTIL>  Minimum utilization percentage (1-100)

Options:
  -d, --device-ids <DEVICE_IDS>  GPU IDs to control, separated by commas
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