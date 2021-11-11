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

//! Recording of results from tests.

pub struct TestResult {
    name: String,
    // TODO: statistics. We're lean and mean for now.
    total_time: f64,
    n_elements: u64,
    status: Status,
}

pub enum Status {
    Pass,
    Fail(String),
    #[allow(unused)]
    Skipped(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReportStyle {
    Short,
    Verbose,
}

impl TestResult {
    pub fn new(name: impl Into<String>) -> TestResult {
        TestResult {
            name: name.into(),
            total_time: 0.0,
            n_elements: 0,
            status: Status::Pass,
        }
    }

    pub fn report(&self, style: ReportStyle) {
        let fail_string = match &self.status {
            Status::Pass => "pass".into(),
            Status::Fail(s) => format!("fail ({})", s),
            Status::Skipped(s) => format!("skipped ({})", s),
        };
        match style {
            ReportStyle::Short => {
                let mut timing_string = String::new();
                if self.total_time > 0.0 {
                    if self.n_elements > 0 {
                        let throughput = self.n_elements as f64 / self.total_time;
                        timing_string = format!(" {} elements/s", format_nice(throughput, 1));
                    } else {
                        timing_string = format!(" {}s", format_nice(self.total_time, 1));
                    }
                }
                println!("{}: {}{}", self.name, fail_string, timing_string)
            }
            ReportStyle::Verbose => {
                println!("test {}", self.name);
                println!("  {}", fail_string);
                if self.total_time > 0.0 {
                    println!("  {}s total time", format_nice(self.total_time, 1));
                    if self.n_elements > 0 {
                        println!("  {} elements", self.n_elements);
                        let throughput = self.n_elements as f64 / self.total_time;
                        println!("  {} elements/s", format_nice(throughput, 1));
                    }
                }
            }
        }
    }

    pub fn fail(&mut self, explanation: impl Into<String>) {
        self.status = Status::Fail(explanation.into());
    }

    #[allow(unused)]
    pub fn skip(&mut self, explanation: impl Into<String>) {
        self.status = Status::Skipped(explanation.into());
    }

    pub fn timing(&mut self, total_time: f64, n_elements: u64) {
        self.total_time = total_time;
        self.n_elements = n_elements;
    }
}

fn format_nice(x: f64, precision: usize) -> String {
    // Precision should probably scale; later
    let (scale, suffix) = if x >= 1e12 && x < 1e15 {
        (1e-12, "T")
    } else if x >= 1e9 {
        (1e-9, "G")
    } else if x >= 1e6 {
        (1e-6, "M")
    } else if x >= 1e3 {
        (1e-3, "k")
    } else if x >= 1.0 {
        (1.0, "")
    } else if x >= 1e-3 {
        (1e3, "m")
    } else if x >= 1e-6 {
        (1e6, "\u{00b5}")
    } else if x >= 1e-9 {
        (1e9, "n")
    } else if x >= 1e-12 {
        (1e12, "p")
    } else {
        return format!("{:.*e}", precision, x);
    };
    format!("{:.*}{}", precision, scale * x, suffix)
}
