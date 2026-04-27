use crate::dynlib::{c_string, load_symbol, open_library};
use crate::utils::Signal;
use anyhow::{Context, Result, bail};
use libloading::Library;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const AMD_KERNEL_IMAGE: &[u8] = include_bytes!("hip/kernel.hsaco");
const HIP_SUCCESS: HipError = 0;
const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
const AMDSMI_PROCESSOR_TYPE_AMD_GPU: AmdSmiProcessorType = 1;
const AMDSMI_STATUS_SUCCESS: AmdSmiStatus = 0;
const AMDSMI_STATUS_OUT_OF_RESOURCES: AmdSmiStatus = 15;
const AMDSMI_MAX_STRING_LENGTH: usize = 256;
const AMDSMI_MEM_TYPE_VRAM: AmdSmiMemoryType = 0;

type HipError = c_int;
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type AmdSmiSocketHandle = *mut c_void;
type AmdSmiProcessorHandle = *mut c_void;
type AmdSmiStatus = c_uint;
type AmdSmiProcessorType = c_uint;
type AmdSmiMemoryType = c_uint;

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdSmiEnumerationInfo {
    drm_render: u32,
    drm_card: u32,
    hsa_id: u32,
    hip_id: u32,
    hip_uuid: [c_char; AMDSMI_MAX_STRING_LENGTH],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdSmiEngineUsage {
    gfx_activity: u32,
    umc_activity: u32,
    mm_activity: u32,
    reserved: [u32; 13],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdSmiProcEngineUsage {
    gfx: u64,
    enc: u64,
    reserved: [u32; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdSmiProcMemoryUsage {
    gtt_mem: u64,
    cpu_mem: u64,
    vram_mem: u64,
    reserved: [u32; 10],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AmdSmiProcInfo {
    name: [c_char; AMDSMI_MAX_STRING_LENGTH],
    pid: u32,
    mem: u64,
    engine_usage: AmdSmiProcEngineUsage,
    memory_usage: AmdSmiProcMemoryUsage,
    container_name: [c_char; AMDSMI_MAX_STRING_LENGTH],
    cu_occupancy: u32,
    evicted_time: u32,
    reserved: [u32; 10],
}

#[derive(Clone, Copy)]
pub struct AmdDevice {
    hip_id: u32,
    handle: AmdSmiProcessorHandle,
}

pub struct AmdBackend {
    hip: HipApi,
    smi: AmdSmi,
    devices: Vec<AmdDevice>,
}

unsafe impl Send for AmdBackend {}
unsafe impl Sync for AmdBackend {}

impl AmdBackend {
    pub fn new() -> Result<Self> {
        let hip = HipApi::load().context("Failed to load HIP runtime")?;
        hip.init().context("Failed to initialize HIP runtime")?;
        let hip_device_count = hip
            .device_count()
            .context("Failed to get HIP device count")?;
        if hip_device_count == 0 {
            bail!("No HIP devices were found");
        }

        let smi = AmdSmi::init().context("Failed to initialize AMD SMI")?;

        let mut backend = Self {
            hip,
            smi,
            devices: Vec::new(),
        };
        backend.devices = backend
            .enumerate_devices(hip_device_count)
            .context("Failed to enumerate AMD GPUs")?;
        if backend.devices.is_empty() {
            bail!("No AMD GPUs were found by AMD SMI");
        }
        Ok(backend)
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn device_ids(&self) -> Vec<u32> {
        self.devices.iter().map(|device| device.hip_id).collect()
    }

    fn device(&self, device_id: u32) -> Result<AmdDevice> {
        self.devices
            .iter()
            .copied()
            .find(|device| device.hip_id == device_id)
            .with_context(|| {
                format!(
                    "AMD GPU {device_id} was not found. Available HIP GPU IDs: {}",
                    self.device_ids()
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    fn enumerate_devices(&self, hip_device_count: u32) -> Result<Vec<AmdDevice>> {
        let mut devices = Vec::new();
        for socket in self.smi.socket_handles()? {
            for processor in self.smi.processor_handles(socket)? {
                match self.smi.processor_type(processor) {
                    Ok(processor_type) if processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU => {
                        continue;
                    }
                    Ok(_) => {}
                    Err(e) => log::warn!("Failed to query AMD processor type: {e:#}"),
                }

                let hip_id = match self.smi.gpu_enumeration_info(processor) {
                    Ok(info) => info.hip_id,
                    Err(e) => {
                        let fallback_id = devices.len() as u32;
                        log::warn!(
                            "Failed to query AMD GPU enumeration info, using fallback HIP ID {fallback_id}: {e:#}"
                        );
                        fallback_id
                    }
                };

                if hip_id >= hip_device_count {
                    log::warn!(
                        "Skipping AMD GPU with HIP ID {hip_id}, because HIP reports only {hip_device_count} device(s)"
                    );
                    continue;
                }

                devices.push(AmdDevice {
                    hip_id,
                    handle: processor,
                });
            }
        }
        devices.sort_by_key(|device| device.hip_id);
        devices.dedup_by_key(|device| device.hip_id);
        Ok(devices)
    }

    fn gpu_activity(&self, device: AmdDevice) -> Result<u32> {
        self.smi.gpu_activity(device.handle)
    }

    fn gpu_processes(&self, device: AmdDevice) -> Result<Vec<AmdSmiProcInfo>> {
        self.smi.gpu_processes(device.handle)
    }
}

pub struct AmdGpuController {
    device_id: u32,
    device: AmdDevice,
    window_duration: Duration,
    target_util: u32,
    mem_reserve_bytes: usize,
    signal: Signal,
    backend: Arc<AmdBackend>,
    _module: HipModuleHandle,
    function: HipFunction,
    loops_per_ms: f64,
}

impl AmdGpuController {
    pub fn new(
        device_id: u32,
        target_util: u32,
        mem_reserve_gb: u32,
        signal: Signal,
        backend: Arc<AmdBackend>,
    ) -> Result<Self> {
        let device = backend.device(device_id)?;
        backend
            .hip
            .set_device(device.hip_id)
            .with_context(|| format!("Failed to select AMD GPU {device_id}"))?;

        let module = load_embedded_kernel(backend.clone()).context("Failed to load HIP kernel")?;
        let function = module
            .function("busy_loop")
            .context("Failed to get HIP kernel function")?;
        let loops_per_ms =
            calibrate(backend.clone(), function).context("Failed to calibrate HIP kernel")?;

        Ok(Self {
            device_id,
            device,
            window_duration: Duration::from_millis(100),
            target_util,
            mem_reserve_bytes: mem_reserve_gb as usize * 1024 * 1024 * 1024,
            signal,
            backend,
            _module: module,
            function,
            loops_per_ms,
        })
    }

    pub fn hold_gpu(&self) -> Result<()> {
        self.backend
            .hip
            .set_device(self.device.hip_id)
            .with_context(|| format!("Failed to select AMD GPU {}", self.device_id))?;
        let stream = HipStreamHandle::new(self.backend.clone())?;
        let dummy_data =
            HipDeviceMemory::new_zeroed(self.backend.clone(), std::mem::size_of::<f32>())?;
        let mut self_util = self.target_util;
        let mut last_seen = Instant::now();
        let mut mem_reserve: Option<HipDeviceMemory> = None;

        while self.signal.load(Ordering::Relaxed) {
            let now = Instant::now();
            if now.duration_since(last_seen) >= Duration::from_secs(2) {
                let total_util = match self.backend.gpu_activity(self.device) {
                    Ok(util) => util.min(100),
                    Err(e) => {
                        log::warn!(
                            "Failed to query AMD GPU {} utilization: {e:#}",
                            self.device_id
                        );
                        std::thread::sleep(Duration::from_secs(1));
                        continue;
                    }
                };

                let processes = match self.backend.gpu_processes(self.device) {
                    Ok(processes) => Some(processes),
                    Err(e) => {
                        log::warn!(
                            "Failed to query AMD GPU {} process list: {e:#}",
                            self.device_id
                        );
                        None
                    }
                };

                if self.mem_reserve_bytes > 0 {
                    match processes.as_ref().map(|processes| processes.len()) {
                        Some(count) if count <= 1 => {
                            if mem_reserve.is_none() {
                                match HipDeviceMemory::new(
                                    self.backend.clone(),
                                    self.mem_reserve_bytes,
                                ) {
                                    Ok(buf) => mem_reserve = Some(buf),
                                    Err(e) => log::warn!(
                                        "Failed to reserve {}GB AMD GPU memory: {e:#}",
                                        self.mem_reserve_bytes / 1024 / 1024 / 1024
                                    ),
                                }
                            }
                        }
                        Some(_) => {
                            mem_reserve = None;
                        }
                        None => {}
                    }
                }

                let others_util = total_util.saturating_sub(self_util);
                self_util = self.target_util.saturating_sub(others_util);
                last_seen = now;
            }

            let perturbed_util = if self_util == 0 {
                0u32
            } else {
                let noise = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos()
                    % 11) as i32
                    - 5;
                (self_util as i32 + noise).clamp(0, 100) as u32
            };

            let work_duration_ms =
                (self.window_duration.as_millis() * perturbed_util as u128) / 100;
            let sleep_duration = self
                .window_duration
                .saturating_sub(Duration::from_millis(work_duration_ms as u64));
            let num_loops_for_work = (work_duration_ms as f64 * self.loops_per_ms) as i32;
            if num_loops_for_work > 0 {
                launch_busy_loop(
                    &self.backend,
                    self.function,
                    stream.raw(),
                    dummy_data.ptr(),
                    num_loops_for_work,
                )?;
                stream.synchronize()?;
            }

            if !sleep_duration.is_zero() {
                std::thread::sleep(sleep_duration);
            }
        }

        Ok(())
    }
}

fn calibrate(backend: Arc<AmdBackend>, function: HipFunction) -> Result<f64> {
    let calibration_loops = 1_000_000;
    let stream = HipStreamHandle::new(backend.clone())?;
    let dummy_data = HipDeviceMemory::new_zeroed(backend.clone(), std::mem::size_of::<f32>())?;
    let start = Instant::now();
    launch_busy_loop(
        &backend,
        function,
        stream.raw(),
        dummy_data.ptr(),
        calibration_loops,
    )?;
    stream.synchronize()?;
    let elapsed = start.elapsed();
    if elapsed.is_zero() {
        bail!("HIP calibration finished too quickly to measure");
    }
    Ok(calibration_loops as f64 / elapsed.as_secs_f64() / 1000.0)
}

fn launch_busy_loop(
    backend: &AmdBackend,
    function: HipFunction,
    stream: HipStream,
    data: *mut c_void,
    num_loops: c_int,
) -> Result<()> {
    let mut data_arg = data;
    let mut loops_arg = num_loops;
    let mut params = [
        (&mut data_arg as *mut *mut c_void).cast::<c_void>(),
        (&mut loops_arg as *mut c_int).cast::<c_void>(),
    ];
    backend.hip.module_launch_kernel(
        function,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        stream,
        params.as_mut_ptr(),
        ptr::null_mut(),
    )
}

fn load_embedded_kernel(backend: Arc<AmdBackend>) -> Result<HipModuleHandle> {
    let mut module = ptr::null_mut();
    backend
        .hip
        .module_load_data(&mut module, AMD_KERNEL_IMAGE.as_ptr().cast::<c_void>())
        .context("Failed to load HIP module")?;

    Ok(HipModuleHandle { module, backend })
}

struct HipDeviceMemory {
    ptr: *mut c_void,
    backend: Arc<AmdBackend>,
}

impl HipDeviceMemory {
    fn new(backend: Arc<AmdBackend>, size: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        backend.hip.malloc(&mut ptr, size)?;
        Ok(Self { ptr, backend })
    }

    fn new_zeroed(backend: Arc<AmdBackend>, size: usize) -> Result<Self> {
        let memory = Self::new(backend, size)?;
        memory.backend.hip.memset(memory.ptr, 0, size)?;
        Ok(memory)
    }

    fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for HipDeviceMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Err(e) = self.backend.hip.free(self.ptr) {
                log::warn!("Failed to free HIP device memory: {e:#}");
            }
        }
    }
}

struct HipStreamHandle {
    stream: HipStream,
    backend: Arc<AmdBackend>,
}

impl HipStreamHandle {
    fn new(backend: Arc<AmdBackend>) -> Result<Self> {
        let mut stream = ptr::null_mut();
        backend
            .hip
            .stream_create_with_flags(&mut stream, HIP_STREAM_NON_BLOCKING)?;
        Ok(Self { stream, backend })
    }

    fn raw(&self) -> HipStream {
        self.stream
    }

    fn synchronize(&self) -> Result<()> {
        self.backend.hip.stream_synchronize(self.stream)
    }
}

impl Drop for HipStreamHandle {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            if let Err(e) = self.backend.hip.stream_destroy(self.stream) {
                log::warn!("Failed to destroy HIP stream: {e:#}");
            }
        }
    }
}

struct HipModuleHandle {
    module: HipModule,
    backend: Arc<AmdBackend>,
}

impl HipModuleHandle {
    fn function(&self, name: &str) -> Result<HipFunction> {
        let name = CString::new(name)?;
        let mut function = ptr::null_mut();
        self.backend
            .hip
            .module_get_function(&mut function, self.module, name.as_ptr())?;
        Ok(function)
    }
}

impl Drop for HipModuleHandle {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Err(e) = self.backend.hip.module_unload(self.module) {
                log::warn!("Failed to unload HIP module: {e:#}");
            }
        }
    }
}

struct HipApi {
    _lib: Library,
    hip_init: unsafe extern "C" fn(c_uint) -> HipError,
    hip_get_device_count: unsafe extern "C" fn(*mut c_int) -> HipError,
    hip_set_device: unsafe extern "C" fn(c_int) -> HipError,
    hip_get_error_string: unsafe extern "C" fn(HipError) -> *const c_char,
    hip_stream_create_with_flags: unsafe extern "C" fn(*mut HipStream, c_uint) -> HipError,
    hip_stream_destroy: unsafe extern "C" fn(HipStream) -> HipError,
    hip_stream_synchronize: unsafe extern "C" fn(HipStream) -> HipError,
    hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError,
    hip_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> HipError,
    hip_free: unsafe extern "C" fn(*mut c_void) -> HipError,
    hip_module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError,
    hip_module_unload: unsafe extern "C" fn(HipModule) -> HipError,
    hip_module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
    hip_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipError,
}

unsafe impl Send for HipApi {}
unsafe impl Sync for HipApi {}

impl HipApi {
    fn load() -> Result<Self> {
        let lib = open_library(&[
            "libamdhip64.so",
            "libamdhip64.so.7",
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so.7",
        ])?;
        Ok(Self {
            hip_init: load_symbol(&lib, b"hipInit\0")?,
            hip_get_device_count: load_symbol(&lib, b"hipGetDeviceCount\0")?,
            hip_set_device: load_symbol(&lib, b"hipSetDevice\0")?,
            hip_get_error_string: load_symbol(&lib, b"hipGetErrorString\0")?,
            hip_stream_create_with_flags: load_symbol(&lib, b"hipStreamCreateWithFlags\0")?,
            hip_stream_destroy: load_symbol(&lib, b"hipStreamDestroy\0")?,
            hip_stream_synchronize: load_symbol(&lib, b"hipStreamSynchronize\0")?,
            hip_malloc: load_symbol(&lib, b"hipMalloc\0")?,
            hip_memset: load_symbol(&lib, b"hipMemset\0")?,
            hip_free: load_symbol(&lib, b"hipFree\0")?,
            hip_module_load_data: load_symbol(&lib, b"hipModuleLoadData\0")?,
            hip_module_unload: load_symbol(&lib, b"hipModuleUnload\0")?,
            hip_module_get_function: load_symbol(&lib, b"hipModuleGetFunction\0")?,
            hip_module_launch_kernel: load_symbol(&lib, b"hipModuleLaunchKernel\0")?,
            _lib: lib,
        })
    }

    fn init(&self) -> Result<()> {
        self.check(unsafe { (self.hip_init)(0) }, "hipInit")
    }

    fn device_count(&self) -> Result<u32> {
        let mut count = 0;
        self.check(
            unsafe { (self.hip_get_device_count)(&mut count) },
            "hipGetDeviceCount",
        )?;
        Ok(count as u32)
    }

    fn set_device(&self, device_id: u32) -> Result<()> {
        self.check(
            unsafe { (self.hip_set_device)(device_id as c_int) },
            "hipSetDevice",
        )
    }

    fn stream_create_with_flags(&self, stream: *mut HipStream, flags: c_uint) -> Result<()> {
        self.check(
            unsafe { (self.hip_stream_create_with_flags)(stream, flags) },
            "hipStreamCreateWithFlags",
        )
    }

    fn stream_destroy(&self, stream: HipStream) -> Result<()> {
        self.check(
            unsafe { (self.hip_stream_destroy)(stream) },
            "hipStreamDestroy",
        )
    }

    fn stream_synchronize(&self, stream: HipStream) -> Result<()> {
        self.check(
            unsafe { (self.hip_stream_synchronize)(stream) },
            "hipStreamSynchronize",
        )
    }

    fn malloc(&self, ptr: *mut *mut c_void, size: usize) -> Result<()> {
        self.check(unsafe { (self.hip_malloc)(ptr, size) }, "hipMalloc")
    }

    fn memset(&self, ptr: *mut c_void, value: c_int, size: usize) -> Result<()> {
        self.check(unsafe { (self.hip_memset)(ptr, value, size) }, "hipMemset")
    }

    fn free(&self, ptr: *mut c_void) -> Result<()> {
        self.check(unsafe { (self.hip_free)(ptr) }, "hipFree")
    }

    fn module_load_data(&self, module: *mut HipModule, image: *const c_void) -> Result<()> {
        self.check(
            unsafe { (self.hip_module_load_data)(module, image) },
            "hipModuleLoadData",
        )
    }

    fn module_unload(&self, module: HipModule) -> Result<()> {
        self.check(
            unsafe { (self.hip_module_unload)(module) },
            "hipModuleUnload",
        )
    }

    fn module_get_function(
        &self,
        function: *mut HipFunction,
        module: HipModule,
        name: *const c_char,
    ) -> Result<()> {
        self.check(
            unsafe { (self.hip_module_get_function)(function, module, name) },
            "hipModuleGetFunction",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn module_launch_kernel(
        &self,
        function: HipFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> Result<()> {
        self.check(
            unsafe {
                (self.hip_module_launch_kernel)(
                    function,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    stream,
                    kernel_params,
                    extra,
                )
            },
            "hipModuleLaunchKernel",
        )
    }

    fn check(&self, status: HipError, context: &str) -> Result<()> {
        if status == HIP_SUCCESS {
            return Ok(());
        }
        bail!("{context} failed: {}", self.error_string(status));
    }

    fn error_string(&self, status: HipError) -> String {
        let ptr = unsafe { (self.hip_get_error_string)(status) };
        c_string(ptr)
    }
}

struct AmdSmi {
    _lib: Library,
    amdsmi_shut_down: unsafe extern "C" fn() -> AmdSmiStatus,
    amdsmi_get_socket_handles:
        unsafe extern "C" fn(*mut u32, *mut AmdSmiSocketHandle) -> AmdSmiStatus,
    amdsmi_get_processor_handles: unsafe extern "C" fn(
        AmdSmiSocketHandle,
        *mut u32,
        *mut AmdSmiProcessorHandle,
    ) -> AmdSmiStatus,
    amdsmi_get_processor_type:
        unsafe extern "C" fn(AmdSmiProcessorHandle, *mut AmdSmiProcessorType) -> AmdSmiStatus,
    amdsmi_get_gpu_enumeration_info:
        unsafe extern "C" fn(AmdSmiProcessorHandle, *mut AmdSmiEnumerationInfo) -> AmdSmiStatus,
    amdsmi_get_gpu_activity:
        unsafe extern "C" fn(AmdSmiProcessorHandle, *mut AmdSmiEngineUsage) -> AmdSmiStatus,
    amdsmi_get_gpu_process_list:
        unsafe extern "C" fn(AmdSmiProcessorHandle, *mut u32, *mut AmdSmiProcInfo) -> AmdSmiStatus,
    amdsmi_get_gpu_memory_usage:
        unsafe extern "C" fn(AmdSmiProcessorHandle, AmdSmiMemoryType, *mut u64) -> AmdSmiStatus,
}

unsafe impl Send for AmdSmi {}
unsafe impl Sync for AmdSmi {}

impl AmdSmi {
    fn init() -> Result<Self> {
        let lib = open_library(&[
            "libamd_smi.so",
            "libamd_smi.so.26",
            "/opt/rocm/lib/libamd_smi.so",
            "/opt/rocm/lib/libamd_smi.so.26",
        ])?;
        let amdsmi_init: unsafe extern "C" fn(u64) -> AmdSmiStatus =
            load_symbol(&lib, b"amdsmi_init\0")?;
        let smi = Self {
            amdsmi_shut_down: load_symbol(&lib, b"amdsmi_shut_down\0")?,
            amdsmi_get_socket_handles: load_symbol(&lib, b"amdsmi_get_socket_handles\0")?,
            amdsmi_get_processor_handles: load_symbol(&lib, b"amdsmi_get_processor_handles\0")?,
            amdsmi_get_processor_type: load_symbol(&lib, b"amdsmi_get_processor_type\0")?,
            amdsmi_get_gpu_enumeration_info: load_symbol(
                &lib,
                b"amdsmi_get_gpu_enumeration_info\0",
            )?,
            amdsmi_get_gpu_activity: load_symbol(&lib, b"amdsmi_get_gpu_activity\0")?,
            amdsmi_get_gpu_process_list: load_symbol(&lib, b"amdsmi_get_gpu_process_list\0")?,
            amdsmi_get_gpu_memory_usage: load_symbol(&lib, b"amdsmi_get_gpu_memory_usage\0")?,
            _lib: lib,
        };
        smi.check(unsafe { amdsmi_init(AMDSMI_INIT_AMD_GPUS) }, "amdsmi_init")?;
        Ok(smi)
    }

    fn socket_handles(&self) -> Result<Vec<AmdSmiSocketHandle>> {
        let mut count = 0u32;
        self.check(
            unsafe { (self.amdsmi_get_socket_handles)(&mut count, ptr::null_mut()) },
            "amdsmi_get_socket_handles",
        )?;
        let mut sockets = vec![ptr::null_mut(); count as usize];
        if count == 0 {
            return Ok(sockets);
        }
        self.check(
            unsafe { (self.amdsmi_get_socket_handles)(&mut count, sockets.as_mut_ptr()) },
            "amdsmi_get_socket_handles",
        )?;
        sockets.truncate(count as usize);
        Ok(sockets)
    }

    fn processor_handles(&self, socket: AmdSmiSocketHandle) -> Result<Vec<AmdSmiProcessorHandle>> {
        let mut count = 0u32;
        self.check(
            unsafe { (self.amdsmi_get_processor_handles)(socket, &mut count, ptr::null_mut()) },
            "amdsmi_get_processor_handles",
        )?;
        let mut processors = vec![ptr::null_mut(); count as usize];
        if count == 0 {
            return Ok(processors);
        }
        self.check(
            unsafe {
                (self.amdsmi_get_processor_handles)(socket, &mut count, processors.as_mut_ptr())
            },
            "amdsmi_get_processor_handles",
        )?;
        processors.truncate(count as usize);
        Ok(processors)
    }

    fn processor_type(&self, processor: AmdSmiProcessorHandle) -> Result<AmdSmiProcessorType> {
        let mut processor_type = 0;
        self.check(
            unsafe { (self.amdsmi_get_processor_type)(processor, &mut processor_type) },
            "amdsmi_get_processor_type",
        )?;
        Ok(processor_type)
    }

    fn gpu_enumeration_info(
        &self,
        processor: AmdSmiProcessorHandle,
    ) -> Result<AmdSmiEnumerationInfo> {
        let mut info = unsafe { std::mem::zeroed::<AmdSmiEnumerationInfo>() };
        self.check(
            unsafe { (self.amdsmi_get_gpu_enumeration_info)(processor, &mut info) },
            "amdsmi_get_gpu_enumeration_info",
        )?;
        Ok(info)
    }

    fn gpu_activity(&self, processor: AmdSmiProcessorHandle) -> Result<u32> {
        let mut info = unsafe { std::mem::zeroed::<AmdSmiEngineUsage>() };
        self.check(
            unsafe { (self.amdsmi_get_gpu_activity)(processor, &mut info) },
            "amdsmi_get_gpu_activity",
        )?;
        Ok(info.gfx_activity)
    }

    fn gpu_processes(&self, processor: AmdSmiProcessorHandle) -> Result<Vec<AmdSmiProcInfo>> {
        let mut max_processes = 0u32;
        let status = unsafe {
            (self.amdsmi_get_gpu_process_list)(processor, &mut max_processes, ptr::null_mut())
        };
        if status != AMDSMI_STATUS_SUCCESS && status != AMDSMI_STATUS_OUT_OF_RESOURCES {
            self.check(status, "amdsmi_get_gpu_process_list")?;
        }
        if max_processes == 0 {
            return Ok(Vec::new());
        }

        let mut processes =
            vec![unsafe { std::mem::zeroed::<AmdSmiProcInfo>() }; max_processes as usize];
        let status = unsafe {
            (self.amdsmi_get_gpu_process_list)(
                processor,
                &mut max_processes,
                processes.as_mut_ptr(),
            )
        };
        if status != AMDSMI_STATUS_SUCCESS && status != AMDSMI_STATUS_OUT_OF_RESOURCES {
            self.check(status, "amdsmi_get_gpu_process_list")?;
        }
        processes.truncate(max_processes as usize);
        Ok(processes)
    }

    #[allow(dead_code)]
    fn gpu_memory_usage(&self, processor: AmdSmiProcessorHandle) -> Result<u64> {
        let mut used = 0u64;
        self.check(
            unsafe {
                (self.amdsmi_get_gpu_memory_usage)(processor, AMDSMI_MEM_TYPE_VRAM, &mut used)
            },
            "amdsmi_get_gpu_memory_usage",
        )?;
        Ok(used)
    }

    fn check(&self, status: AmdSmiStatus, context: &str) -> Result<()> {
        if status == AMDSMI_STATUS_SUCCESS {
            return Ok(());
        }
        bail!("{context} failed with AMD SMI status {status}");
    }
}

impl Drop for AmdSmi {
    fn drop(&mut self) {
        let status = unsafe { (self.amdsmi_shut_down)() };
        if status != AMDSMI_STATUS_SUCCESS {
            log::warn!("amdsmi_shut_down failed with AMD SMI status {status}");
        }
    }
}
