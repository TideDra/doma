use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
mod amd_controller;
mod dynlib;
mod nvidia_controller;
mod utils;
use env_logger::{Builder, Target};
use nvidia_controller::{CudaApi, GPUController};
use utils::{Signal, new_signal};

const AMD_HEALTH_CHECK_DEVICE_ENV: &str = "DOMA_AMD_HEALTH_CHECK_DEVICE";
const AMD_LIST_DEVICES_ENV: &str = "DOMA_AMD_LIST_DEVICES";
const AMD_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

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

    match amd_device_ids() {
        Ok(device_ids) if !device_ids.is_empty() => Ok(Backend::Amd),
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
    let available_device_ids = amd_device_ids()?;
    let mut device_ids = args
        .device_ids
        .clone()
        .unwrap_or_else(|| available_device_ids.clone());
    validate_amd_device_ids(&available_device_ids, &device_ids)?;
    device_ids = healthy_amd_device_ids(device_ids)?;
    let backend = Arc::new(amd_controller::AmdBackend::new()?);
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

fn amd_device_ids() -> Result<Vec<u32>> {
    let exe = std::env::current_exe().context("Failed to get current executable path")?;
    let output = Command::new(exe)
        .env(AMD_LIST_DEVICES_ENV, "1")
        .stdin(Stdio::null())
        .output()
        .context("Failed to enumerate AMD GPUs")?;
    if !output.status.success() {
        bail!(
            "Failed to enumerate AMD GPUs: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout = String::from_utf8(output.stdout).context("AMD GPU list is not valid UTF-8")?;
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return Ok(Vec::new());
    }

    stdout
        .split(',')
        .map(|id| {
            id.parse::<u32>()
                .with_context(|| format!("AMD GPU ID '{id}' is not a valid number"))
        })
        .collect()
}

fn validate_amd_device_ids(available: &[u32], device_ids: &[u32]) -> Result<()> {
    let missing = device_ids
        .iter()
        .copied()
        .filter(|id| !available.contains(id))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "AMD GPU ID(s) {} were not found. Available HIP GPU IDs: {}",
        missing
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        available
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn healthy_amd_device_ids(device_ids: Vec<u32>) -> Result<Vec<u32>> {
    let exe = std::env::current_exe().context("Failed to get current executable path")?;
    let mut healthy = Vec::new();
    for device_id in device_ids {
        match amd_device_health_check(&exe, device_id)? {
            None => healthy.push(device_id),
            Some(reason) => {
                log::warn!("Skipping AMD GPU {device_id}: HIP health check failed ({reason})");
            }
        }
    }

    if healthy.is_empty() {
        bail!("No selected AMD GPUs passed the HIP health check");
    }
    Ok(healthy)
}

fn amd_device_health_check(exe: &std::path::Path, device_id: u32) -> Result<Option<String>> {
    let mut child = Command::new(exe)
        .env(AMD_HEALTH_CHECK_DEVICE_ENV, device_id.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to start AMD GPU {device_id} health check"))?;

    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return if status.success() {
                Ok(None)
            } else {
                Ok(Some(exit_status_reason(status)))
            };
        }

        if started.elapsed() >= AMD_HEALTH_CHECK_TIMEOUT {
            child.kill()?;
            let _ = child.wait();
            return Ok(Some(format!(
                "timed out after {}s",
                AMD_HEALTH_CHECK_TIMEOUT.as_secs()
            )));
        }

        std::thread::sleep(Duration::from_millis(50));
    }
}

fn exit_status_reason(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("terminated by signal {signal}");
        }
    }

    status.to_string()
}

fn wait_for_ctrl_c_and_join(
    signals: Vec<Signal>,
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

fn run_amd_health_check_if_requested() -> Result<bool> {
    let Some(device_id) = std::env::var_os(AMD_HEALTH_CHECK_DEVICE_ENV) else {
        return Ok(false);
    };
    let device_id = device_id
        .to_str()
        .context("AMD health check device ID is not valid UTF-8")?
        .parse::<u32>()
        .context("AMD health check device ID is not a valid number")?;
    amd_controller::health_check_device(device_id)?;
    Ok(true)
}

fn run_amd_device_list_if_requested() -> Result<bool> {
    if std::env::var_os(AMD_LIST_DEVICES_ENV).is_none() {
        return Ok(false);
    }

    let backend = amd_controller::AmdBackend::new()?;
    println!(
        "{}",
        backend
            .device_ids()
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(true)
}

fn main() -> Result<()> {
    if run_amd_health_check_if_requested()? {
        return Ok(());
    }
    if run_amd_device_list_if_requested()? {
        return Ok(());
    }

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
