use anyhow::Result;
use cust::prelude::*;
use clap::Parser;
use std::sync::atomic::Ordering;
mod gpu_controller;
use gpu_controller::{GPUController, new_signal};
use env_logger::{Builder, Target};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Minimum utilization percentage
    #[arg(value_parser=clap::value_parser!(u32).range(1..=100))]
    min_util: u32,
    /// GPU IDs to control, separated by commas
    #[arg(short, long, value_delimiter = ',')]
    device_ids: Option<Vec<u32>>,
}


fn main() -> Result<()> {
    let mut builder = Builder::from_default_env();
    builder.target(Target::Stdout);
    builder.filter_level(log::LevelFilter::Info); // Set default to Info level
    builder.init();
    let args = Args::parse();
    cust::init(CudaFlags::empty())?;
    let num_devices = cust::device::Device::num_devices()?;
    let device_ids = args.device_ids.unwrap_or((0..num_devices).collect());
    log::info!("Control the minimum utilization of the following GPUs to {}%: {}", args.min_util, device_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "));
    let mut handles = Vec::new();
    let mut signals = Vec::new();
    let nvml = Arc::new(nvml_wrapper::Nvml::init()?);
    for i in device_ids {
        let signal = new_signal();
        signals.push(signal.clone());
        let nvml_clone = nvml.clone();
        let handle =std::thread::spawn(move || {
            let gpu_controller = GPUController::new(i, args.min_util, signal, nvml_clone).unwrap();
            let result = gpu_controller.hold_gpu();
            if let Err(e) = result {
                log::error!("Failed to hold GPU {}: {}", i, e);
            }
        });
        handles.push(handle);
    }
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
