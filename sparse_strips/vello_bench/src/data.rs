// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[derive(Copy, Clone, Debug)]
pub struct DataItem {
    pub name: &'static str,
    pub width: u16,
    pub height: u16,
}

pub const GHOSTSCRIPT_TIGER: DataItem = DataItem {
    name: "gs",
    width: 900,
    height: 900,
};

pub const COAT_OF_ARMS: DataItem = DataItem {
    name: "coat_of_arms",
    width: 917,
    height: 1100,
};

pub const PARIS_30K: DataItem = DataItem {
    name: "paris_30k",
    width: 5269,
    height: 3593,
};
