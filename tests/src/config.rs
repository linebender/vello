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

//! Test config parameters.

use clap::ArgMatches;

pub struct Config {
    pub groups: Groups,
    pub size: Size,
    pub n_iter: u64,
}

pub struct Groups(String);

pub enum Size {
    Small,
    Medium,
    Large,
}

impl Config {
    pub fn from_matches(matches: &ArgMatches) -> Config {
        let groups = Groups::from_str(matches.value_of("groups").unwrap_or("all"));
        let size = Size::from_str(matches.value_of("size").unwrap_or("m"));
        let n_iter = matches
            .value_of("n_iter")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        Config {
            groups,
            size,
            n_iter,
        }
    }
}

impl Groups {
    pub fn from_str(s: &str) -> Groups {
        Groups(s.to_string())
    }

    pub fn matches(&self, group_name: &str) -> bool {
        self.0 == "all" || self.0 == group_name
    }
}

impl Size {
    fn from_str(s: &str) -> Size {
        if s == "small" || s == "s" {
            Size::Small
        } else if s == "large" || s == "l" {
            Size::Large
        } else {
            Size::Medium
        }
    }

    pub fn choose<T>(&self, small: T, medium: T, large: T) -> T {
        match self {
            Size::Small => small,
            Size::Medium => medium,
            Size::Large => large,
        }
    }
}
