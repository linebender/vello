// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Winit example.

use anyhow::Result;

fn main() -> Result<()> {
    #[cfg(not(any(target_os = "android", target_env = "ohos")))]
    {
        with_winit::main()
    }
    #[cfg(any(target_os = "android", target_env = "ohos"))]
    unreachable!()
}
