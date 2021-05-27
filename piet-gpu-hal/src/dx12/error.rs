// Copyright © 2019 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is a Windows-specific error mechanism (adapted from piet-dx12),
//! but we should adapt it to be more general.

use winapi::shared::winerror;

pub enum Error {
    Hresult(winerror::HRESULT),
    ExplainedHr(&'static str, winerror::HRESULT),
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Hresult(hr) => write!(f, "hresult {:x}", hr),
            Error::ExplainedHr(exp, hr) => {
                write!(f, "{}: ", exp)?;
                write_hr(f, *hr)
            }
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {}

/// Strings for errors we're likely to see.
///
/// See https://docs.microsoft.com/en-us/windows/win32/direct3ddxgi/dxgi-error
fn err_str_for_hr(hr: winerror::HRESULT) -> Option<&'static str> {
    Some(match hr as u32 {
        0x80004005 => "E_FAIL",
        0x80070057 => "E_INVALIDARG",
        0x887a0001 => "DXGI_ERROR_INVALID_CALL",
        0x887a0002 => "DXGI_ERROR_NOT_FOUND",
        0x887a0004 => "DXGI_ERROR_UNSUPPORTED",
        0x887a0005 => "DXGI_ERROR_DEVICE_REMOVED",
        0x887a0006 => "DXGI_ERROR_DEVICE_HUNG",
        _ => return None,
    })
}

fn write_hr(f: &mut std::fmt::Formatter, hr: winerror::HRESULT) -> std::fmt::Result {
    if let Some(err_str) = err_str_for_hr(hr) {
        write!(f, "{:x} ({})", hr, err_str)
    } else {
        write!(f, "{:x}", hr)
    }
}

pub type D3DResult<T> = (T, winerror::HRESULT);

pub fn error_if_failed_else_value<T>(result: D3DResult<T>) -> Result<T, Error> {
    let (result_value, hresult) = result;

    if winerror::SUCCEEDED(hresult) {
        Ok(result_value)
    } else {
        Err(Error::Hresult(hresult))
    }
}

pub fn error_if_failed_else_unit(hresult: winerror::HRESULT) -> Result<(), Error> {
    error_if_failed_else_value(((), hresult))
}

pub fn explain_error(hresult: winerror::HRESULT, explanation: &'static str) -> Result<(), Error> {
    if winerror::SUCCEEDED(hresult) {
        Ok(())
    } else {
        Err(Error::ExplainedHr(explanation, hresult))
    }
}
