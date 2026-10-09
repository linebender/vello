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
        let features = [
            "blurred_rounded_rect",
            "image_bicubic",
            "gradient_sweep",
            "external_texture_ycbcr",
        ];
        for mask in [0b0000, 0b0001, 0b0010, 0b0100, 0b1000, 0b1111] {
            let mut compiler = Wesl::new("shaders");
            compiler.use_stripping(true);
            for (bit, feature) in features.iter().enumerate() {
                compiler.set_feature(feature, mask & (1 << bit) != 0);
            }
            let source = compiler
                .compile(&"package::render".parse().unwrap())
                .expect("render shader links")
                .to_string();
            let module = wgsl::parse_str(&source).expect("linked WGSL parses");
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .expect("linked WGSL validates");
            #[cfg(feature = "glsl")]
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

#[cfg(all(test, feature = "glsl"))]
mod tests {
    use std::collections::BTreeSet;

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

    // The WebGL backend assigns a texture unit to each of these samplers and no others. Any extra
    // sampler would default to unit 0 and clash with the integer alphas sampler bound there.
    #[test]
    fn render_glsl_declares_only_samplers_bound_by_webgl() {
        use crate::render::{FRAGMENT_SOURCE, VERTEX_SOURCE, fragment, vertex};

        assert_eq!(
            sampler_uniforms(VERTEX_SOURCE),
            BTreeSet::from([vertex::ENCODED_PAINTS_TEXTURE])
        );
        assert_eq!(
            sampler_uniforms(FRAGMENT_SOURCE),
            BTreeSet::from([
                fragment::ALPHAS_TEXTURE,
                fragment::LAYER_INPUT_TEXTURE,
                fragment::ENCODED_PAINTS_TEXTURE,
                fragment::GRADIENT_TEXTURE,
                fragment::EXTERNAL_TEXTURE_0,
            ])
        );
    }

    fn sampler_uniforms(glsl: &str) -> BTreeSet<&str> {
        glsl.split(';')
            .filter_map(|statement| {
                let tokens = statement.split_whitespace().collect::<Vec<_>>();
                let is_sampler = tokens.first() == Some(&"uniform")
                    && tokens.iter().any(|token| token.contains("sampler"));
                is_sampler.then(|| *tokens.last().unwrap())
            })
            .collect()
    }
}
