use cust::prelude::*;
use anyhow::{Error, Result, Context};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration,Instant,SystemTime,UNIX_EPOCH};

const KERBEL_PTX: &str = include_str!("cuda/kernel.ptx");
//static KERNEL_MODULE: LazyLock<Module> = LazyLock::new(|| Module::from_ptx(KERBEL_PTX, &[]).unwrap());
//static KERNEL_FUNCTION: LazyLock<Function> = LazyLock::new(|| KERNEL_MODULE.get_function("busy_loop").unwrap());

pub type Signal = Arc<AtomicBool>;

pub fn new_signal() -> Signal {
    Arc::new(AtomicBool::new(true))
}

pub struct GPUController {
    device_id: u32,
    window_duration: Duration,
    target_util: u32,
    signal: Signal,
    #[allow(unused)]
    context: cust::context::Context,
    module: Module,
    nvml: Arc<nvml_wrapper::Nvml>,
    loops_per_ms: f64,
}

fn calibrate(function: &Function<'_>) -> Result<f64> {
    let calibration_loops = 1_000_000; // 一个固定的、较大的循环次数用于测试
    let dummy_data = DeviceBuffer::from_slice(&[1.0f32])?;
    let start = Instant::now();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    unsafe {
        launch!(
            function<<<1, 1, 0, stream>>>(
                dummy_data.as_device_ptr(),
                calibration_loops
            )
        )?;
    }
    stream.synchronize()?;
    let elapsed = start.elapsed();
    // 计算每毫秒需要多少次循环
    let loops_per_ms = calibration_loops as f64 / elapsed.as_secs_f64() / 1000.0;
    Ok(loops_per_ms)
}

impl GPUController {
    pub fn new(device_id: u32, target_util: u32, signal: Signal, nvml: Arc<nvml_wrapper::Nvml>) -> Result<Self> {
        //cust::init(CudaFlags::empty()).with_context(|| "Failed to initialize CUDA")?;
        let device = cust::device::Device::get_device(device_id)?;
        let context = cust::context::Context::new(device).with_context(|| "Failed to create context")?;
        let module = Module::from_ptx(KERBEL_PTX, &[]).with_context(|| "Failed to create module")?;
        let function = module.get_function("busy_loop").with_context(|| "Failed to get function")?;
        let loops_per_ms = calibrate(&function).with_context(|| "Failed to calibrate")?;
        let window_duration: Duration = Duration::from_millis(100); // 时间窗口：100ms

        //let nvml = nvml_wrapper::Nvml::init()?;

        Ok(Self {
            device_id,
            window_duration,
            target_util,
            signal,
            context,
            module,
            nvml,
            loops_per_ms,
        })
    }

    pub fn hold_gpu(&self) -> Result<(),Error> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let dummy_data = DeviceBuffer::from_slice(&[1.0f32])?;
        let func = self.module.get_function("busy_loop").with_context(|| "Failed to get function")?;
        let nvml_device = self.nvml.device_by_index(self.device_id).with_context(|| "Failed to get NVML device")?;
        let mut self_util = self.target_util;
        let mut last_seen_timestamp = SystemTime::now();
        while self.signal.load(Ordering::Relaxed) {
            let toc = SystemTime::now();
            if toc.duration_since(last_seen_timestamp).unwrap_or_default().as_secs() >= 2 {
                let timestamp_micros = last_seen_timestamp.duration_since(UNIX_EPOCH)
                    .with_context(|| "Failed to get Unix timestamp")?
                    .as_micros() as u64;
                let process_utilization_stats = nvml_device.process_utilization_stats(Some(timestamp_micros));
                if process_utilization_stats.is_err() {
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
                let sum_of_util = process_utilization_stats.unwrap().iter().map(|stat| stat.sm_util).sum::<u32>();
                let others_util = sum_of_util.saturating_sub(self_util);
                self_util = self.target_util.saturating_sub(others_util);
                last_seen_timestamp = toc;
            }


            let work_duration_ms = (self.window_duration.as_millis() * self_util as u128) / 100;
            let sleep_duration = self.window_duration.saturating_sub(Duration::from_millis(work_duration_ms as u64));
            // 根据工作时长计算内核循环次数
            let num_loops_for_work = (work_duration_ms as f64 * self.loops_per_ms) as i32;
            if num_loops_for_work > 0 {
                unsafe {
                    launch!(func<<<1, 1, 0, stream>>>(dummy_data.as_device_ptr(), num_loops_for_work))?;
                }
                stream.synchronize()?;
            }

            if !sleep_duration.is_zero() {
                std::thread::sleep(sleep_duration);
            }
        }
        Ok(())
    }
}