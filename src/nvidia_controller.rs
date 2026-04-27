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

const KERNEL_PTX: &str = include_str!("cuda/kernel.ptx");
const CUDA_SUCCESS: CudaResult = 0;
const CU_STREAM_NON_BLOCKING: c_uint = 0x01;

type CudaResult = c_int;
type CudaDevice = c_int;
type CudaDevicePtr = u64;
type CudaContext = *mut c_void;
type CudaModule = *mut c_void;
type CudaFunction = *mut c_void;
type CudaStream = *mut c_void;

pub struct CudaApi {
    _lib: Library,
    cu_init: unsafe extern "C" fn(c_uint) -> CudaResult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CudaResult,
    cu_device_get: unsafe extern "C" fn(*mut CudaDevice, c_int) -> CudaResult,
    cu_ctx_create: unsafe extern "C" fn(*mut CudaContext, c_uint, CudaDevice) -> CudaResult,
    cu_ctx_destroy: unsafe extern "C" fn(CudaContext) -> CudaResult,
    cu_module_load_data: unsafe extern "C" fn(*mut CudaModule, *const c_void) -> CudaResult,
    cu_module_unload: unsafe extern "C" fn(CudaModule) -> CudaResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CudaFunction, CudaModule, *const c_char) -> CudaResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CudaDevicePtr, usize) -> CudaResult,
    cu_mem_free: unsafe extern "C" fn(CudaDevicePtr) -> CudaResult,
    cu_memcpy_htod: unsafe extern "C" fn(CudaDevicePtr, *const c_void, usize) -> CudaResult,
    cu_stream_create: unsafe extern "C" fn(*mut CudaStream, c_uint) -> CudaResult,
    cu_stream_destroy: unsafe extern "C" fn(CudaStream) -> CudaResult,
    cu_stream_synchronize: unsafe extern "C" fn(CudaStream) -> CudaResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CudaFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CudaStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CudaResult,
    cu_get_error_string: unsafe extern "C" fn(CudaResult, *mut *const c_char) -> CudaResult,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

impl CudaApi {
    pub fn load() -> Result<Self> {
        let lib = open_library(&[
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib64/libcuda.so",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/local/cuda/compat/libcuda.so.1",
            "/usr/local/cuda/compat/libcuda.so",
        ])?;
        Ok(Self {
            cu_init: load_symbol(&lib, b"cuInit\0")?,
            cu_device_get_count: load_symbol(&lib, b"cuDeviceGetCount\0")?,
            cu_device_get: load_symbol(&lib, b"cuDeviceGet\0")?,
            cu_ctx_create: load_symbol(&lib, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy: load_symbol(&lib, b"cuCtxDestroy_v2\0")?,
            cu_module_load_data: load_symbol(&lib, b"cuModuleLoadData\0")?,
            cu_module_unload: load_symbol(&lib, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&lib, b"cuModuleGetFunction\0")?,
            cu_mem_alloc: load_symbol(&lib, b"cuMemAlloc_v2\0")?,
            cu_mem_free: load_symbol(&lib, b"cuMemFree_v2\0")?,
            cu_memcpy_htod: load_symbol(&lib, b"cuMemcpyHtoD_v2\0")?,
            cu_stream_create: load_symbol(&lib, b"cuStreamCreate\0")?,
            cu_stream_destroy: load_symbol(&lib, b"cuStreamDestroy_v2\0")?,
            cu_stream_synchronize: load_symbol(&lib, b"cuStreamSynchronize\0")?,
            cu_launch_kernel: load_symbol(&lib, b"cuLaunchKernel\0")?,
            cu_get_error_string: load_symbol(&lib, b"cuGetErrorString\0")?,
            _lib: lib,
        })
    }

    pub fn init(&self) -> Result<()> {
        self.check(unsafe { (self.cu_init)(0) }, "cuInit")
    }

    pub fn device_count(&self) -> Result<u32> {
        let mut count = 0;
        self.check(
            unsafe { (self.cu_device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        Ok(count as u32)
    }

    fn device(&self, device_id: u32) -> Result<CudaDevice> {
        let mut device = 0;
        self.check(
            unsafe { (self.cu_device_get)(&mut device, device_id as c_int) },
            "cuDeviceGet",
        )?;
        Ok(device)
    }

    fn create_context(&self, device: CudaDevice) -> Result<CudaContext> {
        let mut context = ptr::null_mut();
        self.check(
            unsafe { (self.cu_ctx_create)(&mut context, 0, device) },
            "cuCtxCreate",
        )?;
        Ok(context)
    }

    fn destroy_context(&self, context: CudaContext) -> Result<()> {
        self.check(unsafe { (self.cu_ctx_destroy)(context) }, "cuCtxDestroy")
    }

    fn module_load_data(&self, image: *const c_void) -> Result<CudaModule> {
        let mut module = ptr::null_mut();
        self.check(
            unsafe { (self.cu_module_load_data)(&mut module, image) },
            "cuModuleLoadData",
        )?;
        Ok(module)
    }

    fn module_unload(&self, module: CudaModule) -> Result<()> {
        self.check(unsafe { (self.cu_module_unload)(module) }, "cuModuleUnload")
    }

    fn module_get_function(&self, module: CudaModule, name: *const c_char) -> Result<CudaFunction> {
        let mut function = ptr::null_mut();
        self.check(
            unsafe { (self.cu_module_get_function)(&mut function, module, name) },
            "cuModuleGetFunction",
        )?;
        Ok(function)
    }

    fn mem_alloc(&self, size: usize) -> Result<CudaDevicePtr> {
        let mut ptr = 0;
        self.check(unsafe { (self.cu_mem_alloc)(&mut ptr, size) }, "cuMemAlloc")?;
        Ok(ptr)
    }

    fn mem_free(&self, ptr: CudaDevicePtr) -> Result<()> {
        self.check(unsafe { (self.cu_mem_free)(ptr) }, "cuMemFree")
    }

    fn memcpy_htod(&self, dst: CudaDevicePtr, src: *const c_void, size: usize) -> Result<()> {
        self.check(
            unsafe { (self.cu_memcpy_htod)(dst, src, size) },
            "cuMemcpyHtoD",
        )
    }

    fn stream_create(&self) -> Result<CudaStream> {
        let mut stream = ptr::null_mut();
        self.check(
            unsafe { (self.cu_stream_create)(&mut stream, CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(stream)
    }

    fn stream_destroy(&self, stream: CudaStream) -> Result<()> {
        self.check(
            unsafe { (self.cu_stream_destroy)(stream) },
            "cuStreamDestroy",
        )
    }

    fn stream_synchronize(&self, stream: CudaStream) -> Result<()> {
        self.check(
            unsafe { (self.cu_stream_synchronize)(stream) },
            "cuStreamSynchronize",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kernel(
        &self,
        function: CudaFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CudaStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> Result<()> {
        self.check(
            unsafe {
                (self.cu_launch_kernel)(
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
            "cuLaunchKernel",
        )
    }

    fn check(&self, status: CudaResult, context: &str) -> Result<()> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }
        bail!("{context} failed: {}", self.error_string(status));
    }

    fn error_string(&self, status: CudaResult) -> String {
        let mut ptr = ptr::null();
        let result = unsafe { (self.cu_get_error_string)(status, &mut ptr) };
        if result != CUDA_SUCCESS {
            return format!("CUDA error {status}");
        }
        c_string(ptr)
    }
}

pub struct GPUController {
    device_id: u32,
    window_duration: Duration,
    target_util: u32,
    mem_reserve_bytes: usize,
    signal: Signal,
    cuda: Arc<CudaApi>,
    _module: CudaModuleHandle,
    _context: CudaContextHandle,
    function: CudaFunction,
    nvml: Arc<nvml_wrapper::Nvml>,
    loops_per_ms: f64,
}

impl GPUController {
    pub fn new(
        device_id: u32,
        target_util: u32,
        mem_reserve_gb: u32,
        signal: Signal,
        cuda: Arc<CudaApi>,
        nvml: Arc<nvml_wrapper::Nvml>,
    ) -> Result<Self> {
        let device = cuda.device(device_id)?;
        let context = CudaContextHandle::new(cuda.clone(), device)
            .with_context(|| "Failed to create CUDA context")?;
        let module =
            CudaModuleHandle::new(cuda.clone()).with_context(|| "Failed to load CUDA module")?;
        let function = module
            .function("busy_loop")
            .with_context(|| "Failed to get CUDA function")?;
        let loops_per_ms =
            calibrate(cuda.clone(), function).with_context(|| "Failed to calibrate")?;
        let window_duration = Duration::from_millis(100);

        Ok(Self {
            device_id,
            window_duration,
            target_util,
            mem_reserve_bytes: mem_reserve_gb as usize * 1024 * 1024 * 1024,
            signal,
            cuda,
            _module: module,
            _context: context,
            function,
            nvml,
            loops_per_ms,
        })
    }

    pub fn hold_gpu(&self) -> Result<()> {
        let stream = CudaStreamHandle::new(self.cuda.clone())?;
        let dummy_data = CudaDeviceMemory::new_with_f32(self.cuda.clone(), 1.0)?;
        let nvml_device = self
            .nvml
            .device_by_index(self.device_id)
            .with_context(|| "Failed to get NVML device")?;
        let mut self_util = self.target_util;
        let mut last_seen_timestamp = SystemTime::now();
        let mut mem_reserve: Option<CudaDeviceMemory> = None;
        while self.signal.load(Ordering::Relaxed) {
            let toc = SystemTime::now();
            if toc
                .duration_since(last_seen_timestamp)
                .unwrap_or_default()
                .as_secs()
                >= 2
            {
                let timestamp_micros = last_seen_timestamp
                    .duration_since(UNIX_EPOCH)
                    .with_context(|| "Failed to get Unix timestamp")?
                    .as_micros() as u64;
                let process_utilization_stats =
                    nvml_device.process_utilization_stats(Some(timestamp_micros));
                if process_utilization_stats.is_err() {
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
                match nvml_device.running_compute_processes_count() {
                    Ok(count) if count <= 1 && self.mem_reserve_bytes > 0 => {
                        if mem_reserve.is_none() {
                            match CudaDeviceMemory::new(self.cuda.clone(), self.mem_reserve_bytes) {
                                Ok(buf) => mem_reserve = Some(buf),
                                Err(e) => log::warn!(
                                    "Failed to reserve {}GB GPU memory: {e}",
                                    self.mem_reserve_bytes / 1024 / 1024 / 1024
                                ),
                            }
                        }
                    }
                    Ok(_) => {
                        mem_reserve = None;
                    }
                    Err(e) => {
                        log::warn!("Failed to get running processes count: {e}");
                    }
                }
                let sum_of_util = process_utilization_stats
                    .unwrap()
                    .iter()
                    .map(|stat| stat.sm_util)
                    .sum::<u32>();
                let others_util = sum_of_util.saturating_sub(self_util);
                self_util = self.target_util.saturating_sub(others_util);
                last_seen_timestamp = toc;
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
                    &self.cuda,
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

fn calibrate(cuda: Arc<CudaApi>, function: CudaFunction) -> Result<f64> {
    let calibration_loops = 1_000_000;
    let stream = CudaStreamHandle::new(cuda.clone())?;
    let dummy_data = CudaDeviceMemory::new_with_f32(cuda.clone(), 1.0)?;
    let start = Instant::now();
    launch_busy_loop(
        &cuda,
        function,
        stream.raw(),
        dummy_data.ptr(),
        calibration_loops,
    )?;
    stream.synchronize()?;
    let elapsed = start.elapsed();
    if elapsed.is_zero() {
        bail!("CUDA calibration finished too quickly to measure");
    }
    Ok(calibration_loops as f64 / elapsed.as_secs_f64() / 1000.0)
}

fn launch_busy_loop(
    cuda: &CudaApi,
    function: CudaFunction,
    stream: CudaStream,
    data: CudaDevicePtr,
    num_loops: c_int,
) -> Result<()> {
    let mut data_arg = data;
    let mut loops_arg = num_loops;
    let mut params = [
        (&mut data_arg as *mut CudaDevicePtr).cast::<c_void>(),
        (&mut loops_arg as *mut c_int).cast::<c_void>(),
    ];
    cuda.launch_kernel(
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

struct CudaContextHandle {
    context: CudaContext,
    cuda: Arc<CudaApi>,
}

impl CudaContextHandle {
    fn new(cuda: Arc<CudaApi>, device: CudaDevice) -> Result<Self> {
        let context = cuda.create_context(device)?;
        Ok(Self { context, cuda })
    }
}

impl Drop for CudaContextHandle {
    fn drop(&mut self) {
        if !self.context.is_null() {
            if let Err(e) = self.cuda.destroy_context(self.context) {
                log::warn!("Failed to destroy CUDA context: {e:#}");
            }
        }
    }
}

struct CudaModuleHandle {
    module: CudaModule,
    cuda: Arc<CudaApi>,
}

impl CudaModuleHandle {
    fn new(cuda: Arc<CudaApi>) -> Result<Self> {
        let ptx = CString::new(KERNEL_PTX)?;
        let module = cuda.module_load_data(ptx.as_ptr().cast::<c_void>())?;
        Ok(Self { module, cuda })
    }

    fn function(&self, name: &str) -> Result<CudaFunction> {
        let name = CString::new(name)?;
        self.cuda.module_get_function(self.module, name.as_ptr())
    }
}

impl Drop for CudaModuleHandle {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Err(e) = self.cuda.module_unload(self.module) {
                log::warn!("Failed to unload CUDA module: {e:#}");
            }
        }
    }
}

struct CudaDeviceMemory {
    ptr: CudaDevicePtr,
    cuda: Arc<CudaApi>,
}

impl CudaDeviceMemory {
    fn new(cuda: Arc<CudaApi>, size: usize) -> Result<Self> {
        let ptr = cuda.mem_alloc(size)?;
        Ok(Self { ptr, cuda })
    }

    fn new_with_f32(cuda: Arc<CudaApi>, value: f32) -> Result<Self> {
        let memory = Self::new(cuda, std::mem::size_of::<f32>())?;
        memory.cuda.memcpy_htod(
            memory.ptr,
            (&value as *const f32).cast::<c_void>(),
            std::mem::size_of::<f32>(),
        )?;
        Ok(memory)
    }

    fn ptr(&self) -> CudaDevicePtr {
        self.ptr
    }
}

impl Drop for CudaDeviceMemory {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Err(e) = self.cuda.mem_free(self.ptr) {
                log::warn!("Failed to free CUDA device memory: {e:#}");
            }
        }
    }
}

struct CudaStreamHandle {
    stream: CudaStream,
    cuda: Arc<CudaApi>,
}

impl CudaStreamHandle {
    fn new(cuda: Arc<CudaApi>) -> Result<Self> {
        let stream = cuda.stream_create()?;
        Ok(Self { stream, cuda })
    }

    fn raw(&self) -> CudaStream {
        self.stream
    }

    fn synchronize(&self) -> Result<()> {
        self.cuda.stream_synchronize(self.stream)
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            if let Err(e) = self.cuda.stream_destroy(self.stream) {
                log::warn!("Failed to destroy CUDA stream: {e:#}");
            }
        }
    }
}
