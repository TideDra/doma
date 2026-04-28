use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub type Signal = Arc<AtomicBool>;

pub fn new_signal() -> Signal {
    Arc::new(AtomicBool::new(true))
}

pub fn process_exists(pid: u32) -> std::io::Result<bool> {
    if pid == 0 {
        return Ok(false);
    }

    std::path::Path::new("/proc")
        .join(pid.to_string())
        .try_exists()
}
