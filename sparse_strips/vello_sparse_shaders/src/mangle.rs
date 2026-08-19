// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compact, deterministic names for declarations imported from WESL modules.

use std::collections::HashMap;
use std::sync::Mutex;

use wesl::{Mangler, ModulePath};

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const BASE62: &[u8; 62] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LETTERS: &[u8; 52] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Mangler producing compact names from stable hashes of fully-qualified declarations.
#[derive(Default)]
pub(crate) struct ShortMangler {
    declarations: Mutex<HashMap<String, (ModulePath, String)>>,
}

impl Mangler for ShortMangler {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let qualified_name = format!("{path}::{item}");
        let mangled = format!("w{}", encode_identifier(fnv1a(qualified_name.as_bytes())));
        let declaration = (path.clone(), item.to_owned());
        let mut declarations = self
            .declarations
            .lock()
            .expect("short mangler mutex should not be poisoned");

        if let Some(previous) = declarations.insert(mangled.clone(), declaration.clone())
            && previous != declaration
        {
            panic!(
                "WESL name-mangling collision: `{}` and `{qualified_name}` both map to `{mangled}`",
                format_args!("{}::{}", previous.0, previous.1)
            );
        }

        mangled
    }

    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        self.declarations
            .lock()
            .expect("short mangler mutex should not be poisoned")
            .get(mangled)
            .cloned()
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn encode_identifier(mut value: u64) -> String {
    let mut buffer = [0; 11];
    let mut start = buffer.len() - 1;

    // Naga appends `_` to names ending in digits so that its own numeric suffixes remain
    // unambiguous. Use a letter for the final digit while retaining base62 for the rest.
    buffer[start] = LETTERS[(value % 52) as usize];
    value /= 52;
    while value != 0 {
        start -= 1;
        buffer[start] = BASE62[(value % 62) as usize];
        value /= 62;
    }

    std::str::from_utf8(&buffer[start..])
        .expect("base62 alphabet should be valid UTF-8")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_short_stable_identifier() {
        let path = "package::helpers::image".parse().unwrap();
        let first = ShortMangler::default();
        let second = ShortMangler::default();

        let first_name = first.mangle(&path, "bilinear_sample");
        let second_name = second.mangle(&path, "bilinear_sample");

        assert_eq!(first_name, second_name);
        assert!(first_name.len() <= 12);
        assert!(first_name.starts_with('w'));
        assert!(
            first_name
                .chars()
                .last()
                .is_some_and(|character| character.is_ascii_alphabetic())
        );
        assert!(
            first_name
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
    }

    #[test]
    fn maps_identifiers_back_to_declarations() {
        let path: ModulePath = "package::helpers::gradient".parse().unwrap();
        let mangler = ShortMangler::default();

        let name = mangler.mangle(&path, "sample_gradient_lut");

        assert_eq!(
            mangler.unmangle(&name),
            Some((path, "sample_gradient_lut".to_owned()))
        );
    }

    #[test]
    fn distinguishes_declarations() {
        let path: ModulePath = "package::helpers::image".parse().unwrap();
        let other_path: ModulePath = "package::helpers::gradient".parse().unwrap();
        let mangler = ShortMangler::default();

        assert_ne!(
            mangler.mangle(&path, "sample"),
            mangler.mangle(&other_path, "sample")
        );
        assert_ne!(
            mangler.mangle(&path, "sample"),
            mangler.mangle(&path, "other")
        );
    }
}
