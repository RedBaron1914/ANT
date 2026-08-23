#[cfg(windows)]
use windows::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};

/// Pin current thread affinity to CCD0 cores on Windows.
#[cfg(windows)]
pub fn pin_thread_to_ccd0() {
    unsafe {
        let thread_handle = GetCurrentThread();
        // 0xFFF = 12 logical cores of CCD0 on Ryzen 5900x with SMT
        SetThreadAffinityMask(thread_handle, 0xFFF);
    }
}

#[cfg(not(windows))]
pub fn pin_thread_to_ccd0() {
    // No-op on non-Windows platforms (e.g. Linux / Google Colab)
}
