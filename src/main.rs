use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::sync::atomic::Ordering;
mod amd_controller;
mod dynlib;
mod gpu_controller;
mod signal;
use env_logger::{Builder, Target};
use gpu_controller::{CudaApi, GPUController};
use signal::new_signal;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Auto,
    Nvidia,
    Amd,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Minimum utilization percentage
    #[arg(value_parser=clap::value_parser!(u32).range(1..=100))]
    min_util: u32,
    /// GPU IDs to control, separated by commas
    #[arg(short, long, value_delimiter = ',')]
    device_ids: Option<Vec<u32>>,
    /// GPU memory to reserve (in GB) when other processes are detected
    #[arg(short, long, default_value_t = 0)]
    mem_reserve: u32,
    /// GPU backend to use
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
}

fn nvidia_device_count() -> Result<u32> {
    let cuda = CudaApi::load()?;
    cuda.init()?;
    cuda.device_count()
}

fn detect_backend() -> Result<Backend> {
    match nvidia_device_count() {
        Ok(count) if count > 0 => return Ok(Backend::Nvidia),
        Ok(_) => {}
        Err(e) => log::debug!("NVIDIA backend is not available: {e:#}"),
    }

    match amd_controller::AmdBackend::new() {
        Ok(backend) if backend.device_count() > 0 => Ok(Backend::Amd),
        Ok(_) => anyhow::bail!("No AMD GPUs were found"),
        Err(e) => anyhow::bail!("No supported NVIDIA or AMD GPUs were found: {e:#}"),
    }
}

fn run_nvidia(args: &Args) -> Result<()> {
    let cuda = Arc::new(CudaApi::load()?);
    cuda.init()?;
    let num_devices = cuda.device_count()?;
    let device_ids = args
        .device_ids
        .clone()
        .unwrap_or((0..num_devices).collect());
    log::info!(
        "Control the minimum utilization of the following NVIDIA GPUs to {}%: {}",
        args.min_util,
        device_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut handles = Vec::new();
    let mut signals = Vec::new();
    let nvml = Arc::new(nvml_wrapper::Nvml::init()?);
    for i in device_ids {
        let signal = new_signal();
        signals.push(signal.clone());
        let cuda_clone = cuda.clone();
        let nvml_clone = nvml.clone();
        let min_util = args.min_util;
        let mem_reserve = args.mem_reserve;
        let handle = std::thread::spawn(move || {
            let result = (|| {
                let gpu_controller =
                    GPUController::new(i, min_util, mem_reserve, signal, cuda_clone, nvml_clone)?;
                gpu_controller.hold_gpu()
            })();
            if let Err(e) = result {
                log::error!("Failed to hold NVIDIA GPU {}: {}", i, e);
            }
        });
        handles.push(handle);
    }

    wait_for_ctrl_c_and_join(signals, handles)
}

fn run_amd(args: &Args) -> Result<()> {
    let backend = Arc::new(amd_controller::AmdBackend::new()?);
    let device_ids = args
        .device_ids
        .clone()
        .unwrap_or_else(|| backend.device_ids());
    log::info!(
        "Control the minimum utilization of the following AMD GPUs to {}%: {}",
        args.min_util,
        device_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut handles = Vec::new();
    let mut signals = Vec::new();
    for i in device_ids {
        let signal = new_signal();
        signals.push(signal.clone());
        let backend_clone = backend.clone();
        let min_util = args.min_util;
        let mem_reserve = args.mem_reserve;
        let handle = std::thread::spawn(move || {
            let result = (|| {
                let gpu_controller = amd_controller::AmdGpuController::new(
                    i,
                    min_util,
                    mem_reserve,
                    signal,
                    backend_clone,
                )?;
                gpu_controller.hold_gpu()
            })();
            if let Err(e) = result {
                log::error!("Failed to hold AMD GPU {}: {}", i, e);
            }
        });
        handles.push(handle);
    }

    wait_for_ctrl_c_and_join(signals, handles)
}

fn wait_for_ctrl_c_and_join(
    signals: Vec<signal::Signal>,
    handles: Vec<std::thread::JoinHandle<()>>,
) -> Result<()> {
    ctrlc::set_handler(move || {
        log::info!("Ctrl+C pressed, stopping GPU controllers...");
        for signal in &signals {
            signal.store(false, Ordering::Relaxed);
        }
    })?;

    for handle in handles {
        handle.join().unwrap();
    }
    log::info!("Gracefully stopped GPU controllers.");
    Ok(())
}

fn main() -> Result<()> {
    let mut builder = Builder::from_default_env();
    builder.target(Target::Stdout);
    builder.filter_level(log::LevelFilter::Info); // Set default to Info level
    builder.init();
    let args = Args::parse();
    let backend = match args.backend {
        Backend::Auto => detect_backend()?,
        backend => backend,
    };

    match backend {
        Backend::Auto => unreachable!("auto backend should be resolved before running"),
        Backend::Nvidia => run_nvidia(&args),
        Backend::Amd => run_amd(&args),
    }
}
