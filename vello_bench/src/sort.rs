// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::registry::Registry;

pub fn sort(r: &mut Registry) {
    if !crate::EXTENDED {
        return;
    }

    for item in get_data_items() {
        let unsorted = item.unsorted_tiles();
        r.bench(format!("sort/{}", item.name), move |b| {
            b.iter_batched(|| unsorted.clone(), |mut tiles| tiles.sort_tiles());
        });
    }
}
