use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub type Signal = Arc<AtomicBool>;

pub fn new_signal() -> Signal {
    Arc::new(AtomicBool::new(true))
}
