// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WESL shader sources linked to WGSL and optionally compiled to GLSL for `vello_gpu`.

#[cfg(feature = "glsl")]
mod compile;
#[cfg(feature = "glsl")]
mod lint;
#[cfg(any(test, feature = "glsl"))]
mod minify;
#[cfg(feature = "glsl")]
mod types;

include!(concat!(env!("OUT_DIR"), "/compiled_shaders.rs"));

#[cfg(test)]
mod feature_tests {
    use naga::front::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use wesl::Wesl;

    #[test]
    fn shader_feature_combinations_compile() {
        const FEATURES: [&str; 4] = [
            "blurred_rounded_rect",
            "full_image_support",
            "gradient_sweep",
            "texture_external",
        ];
        let combinations: [&[&str]; 7] = [
            &[],
            &["blurred_rounded_rect"],
            &["full_image_support"],
            &["gradient_sweep"],
            &[
                "blurred_rounded_rect",
                "full_image_support",
                "gradient_sweep",
            ],
            &["texture_external"],
            &FEATURES,
        ];
        for enabled in combinations {
            let mut compiler = Wesl::new("shaders");
            compiler.use_stripping(true);
            for feature in FEATURES {
                compiler.set_feature(feature, enabled.contains(&feature));
            }
            let source = compiler
                .compile(&"package::render".parse().unwrap())
                .expect("render shader links")
                .to_string();
            let module = wgsl::parse_str(&source).expect("linked WGSL parses");
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .expect("linked WGSL validates");
            // GLSL is only generated without `texture_external`, which naga can't translate.
            #[cfg(feature = "glsl")]
            if !enabled.contains(&"texture_external") {
                crate::compile::compile_wgsl_shader(
                    &source,
                    "render",
                    "vs_main",
                    "fs_main",
                    &std::collections::BTreeMap::new(),
                );
            }
        }
    }
}

#[cfg(all(test, feature = "glsl"))]
mod tests {
    use naga::front::wgsl;

    use crate::lint::lint;

    #[test]
    fn every_shipped_shader_passes_the_lint() {
        assert!(
            !crate::wgsl::ALL.is_empty(),
            "expected at least one linked WESL shader"
        );
        for &(name, source) in crate::wgsl::ALL {
            let module = wgsl::parse_str(source).expect("linked WGSL parses");
            lint(name, &module);
        }
    }
}
