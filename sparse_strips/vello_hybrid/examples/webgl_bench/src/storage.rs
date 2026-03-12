// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Save/load benchmark reports to browser localStorage.

use serde::{Deserialize, Serialize};

const STORAGE_KEY: &str = "vello_bench_reports";

/// A single benchmark result within a report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SavedResult {
    pub(crate) name: String,
    pub(crate) ms_per_frame: f64,
    pub(crate) iterations: usize,
}

/// A saved benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BenchReport {
    pub(crate) label: String,
    pub(crate) viewport_width: u32,
    pub(crate) viewport_height: u32,
    pub(crate) results: Vec<SavedResult>,
}

/// All saved reports (stored as a JSON array).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ReportStore {
    pub(crate) reports: Vec<BenchReport>,
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

/// Load all saved reports from localStorage.
pub(crate) fn load_reports() -> ReportStore {
    let Some(storage) = local_storage() else {
        return ReportStore::default();
    };
    let Some(json) = storage.get_item(STORAGE_KEY).ok().flatten() else {
        return ReportStore::default();
    };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Save a report (appends to the list).
pub(crate) fn save_report(report: BenchReport) {
    let mut store = load_reports();
    store.reports.push(report);
    if let Some(storage) = local_storage()
        && let Ok(json) = serde_json::to_string(&store)
    {
        let _ = storage.set_item(STORAGE_KEY, &json);
    }
}

/// Delete a report by index.
pub(crate) fn delete_report(idx: usize) {
    let mut store = load_reports();
    if idx < store.reports.len() {
        store.reports.remove(idx);
        if let Some(storage) = local_storage()
            && let Ok(json) = serde_json::to_string(&store)
        {
            let _ = storage.set_item(STORAGE_KEY, &json);
        }
    }
}
