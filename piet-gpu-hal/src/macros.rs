// Copyright 2021 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Macros, mostly to automate backend selection tedium.

#[doc(hidden)]
/// Configure an item to be included only for the given GPU.
#[macro_export]
macro_rules! mux_cfg {
    ( #[cfg(vk)] $($tokens:tt)* ) => {
        #[cfg(not(target_os="macos"))] $( $tokens )*
    };

    ( #[cfg(dx12)] $($tokens:tt)* ) => {
        #[cfg(target_os="windows")] $( $tokens )*
    };

    ( #[cfg(mtl)] $($tokens:tt)* ) => {
        #[cfg(target_os="macos")] $( $tokens )*
    };
}

#[doc(hidden)]
/// Define an enum with a variant per GPU.
#[macro_export]
macro_rules! mux_enum {
    ( $(#[$outer:meta])* $v:vis enum $name:ident {
        Vk($vk:ty),
        Dx12($dx12:ty),
        Mtl($mtl:ty),
    } ) => {
        $(#[$outer])* $v enum $name {
            #[cfg(not(target_os="macos"))]
            Vk($vk),
            #[cfg(target_os="windows")]
            Dx12($dx12),
            #[cfg(target_os="macos")]
            Mtl($mtl),
        }

        impl $name {
            $crate::mux_cfg! {
                #[cfg(vk)]
                #[allow(unused)]
                fn vk(&self) -> &$vk {
                    match self {
                        $name::Vk(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(vk)]
                #[allow(unused)]
                fn vk_mut(&mut self) -> &mut $vk {
                    match self {
                        $name::Vk(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(vk)]
                #[allow(unused)]
                fn vk_owned(self) -> $vk {
                    match self {
                        $name::Vk(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }

            $crate::mux_cfg! {
                #[cfg(dx12)]
                #[allow(unused)]
                fn dx12(&self) -> &$dx12 {
                    match self {
                        $name::Dx12(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(dx12)]
                #[allow(unused)]
                fn dx12_mut(&mut self) -> &mut $dx12 {
                    match self {
                        $name::Dx12(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(dx12)]
                #[allow(unused)]
                fn dx12_owned(self) -> $dx12 {
                    match self {
                        $name::Dx12(x) => x,
                        _ => panic!("downcast error")
                    }
                }
            }

            $crate::mux_cfg! {
                #[cfg(mtl)]
                #[allow(unused)]
                fn mtl(&self) -> &$mtl {
                    match self {
                        $name::Mtl(x) => x,
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(mtl)]
                #[allow(unused)]
                fn mtl_mut(&mut self) -> &mut $mtl {
                    match self {
                        $name::Mtl(x) => x,
                    }
                }
            }
            $crate::mux_cfg! {
                #[cfg(mtl)]
                #[allow(unused)]
                fn mtl_owned(self) -> $mtl {
                    match self {
                        $name::Mtl(x) => x,
                    }
                }
            }
        }
    };
}

/// Define an enum with a variant per GPU for a Device associated type.
macro_rules! mux_device_enum {
    ( $(#[$outer:meta])* $assoc_type: ident) => {
        $crate::mux_enum! {
            $(#[$outer])*
            pub enum $assoc_type {
                Vk(<$crate::vulkan::VkDevice as $crate::backend::Device>::$assoc_type),
                Dx12(<$crate::dx12::Dx12Device as $crate::backend::Device>::$assoc_type),
                Mtl(<$crate::metal::MtlDevice as $crate::backend::Device>::$assoc_type),
            }
        }
    }
}

#[doc(hidden)]
/// A match statement where match arms are conditionally configured per GPU.
#[macro_export]
macro_rules! mux_match {
    ( $e:expr ;
        $vkname:ident::Vk($vkvar:ident) => $vkblock: block
        $dx12name:ident::Dx12($dx12var:ident) => $dx12block: block
        $mtlname:ident::Mtl($mtlvar:ident) => $mtlblock: block
    ) => {
        match $e {
            #[cfg(not(target_os="macos"))]
            $vkname::Vk($vkvar) => $vkblock
            #[cfg(target_os="windows")]
            $dx12name::Dx12($dx12var) => $dx12block
            #[cfg(target_os="macos")]
            $mtlname::Mtl($mtlvar) => $mtlblock
        }
    };

    ( $e:expr ;
        $vkname:ident::Vk($vkvar:ident) => $vkblock: expr,
        $dx12name:ident::Dx12($dx12var:ident) => $dx12block: expr,
        $mtlname:ident::Mtl($mtlvar:ident) => $mtlblock: expr,
    ) => {
        $crate::mux_match! { $e;
            $vkname::Vk($vkvar) => { $vkblock }
            $dx12name::Dx12($dx12var) => { $dx12block }
            $mtlname::Mtl($mtlvar) => { $mtlblock }
        }
    };
}

/// A convenience macro for selecting a shader from included files.
#[macro_export]
macro_rules! include_shader {
    ( $device:expr, $path_base:expr) => {
        $device.choose_shader(
            include_bytes!(concat!($path_base, ".spv")),
            include_str!(concat!($path_base, ".hlsl")),
            include_str!(concat!($path_base, ".msl")),
        )
    };
}
