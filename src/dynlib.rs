use anyhow::{Context, Result, bail};
use libloading::Library;
use std::ffi::CStr;
use std::os::raw::c_char;

pub fn open_library(candidates: &[&str]) -> Result<Library> {
    let mut errors = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(candidate) } {
            Ok(lib) => return Ok(lib),
            Err(e) => errors.push(format!("{candidate}: {e}")),
        }
    }
    bail!(
        "failed to load any of [{}]: {}",
        candidates.join(", "),
        errors.join("; ")
    )
}

pub fn load_symbol<T: Copy>(lib: &Library, symbol: &[u8]) -> Result<T> {
    let loaded = unsafe { lib.get::<T>(symbol) }
        .with_context(|| format!("Failed to load symbol {}", String::from_utf8_lossy(symbol)))?;
    Ok(*loaded)
}

pub fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "unknown error".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
