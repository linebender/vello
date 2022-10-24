// Copyright 2022 Google LLC
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

use handlebars::Handlebars;
use serde::Serialize;

pub struct ShaderTemplate {
    handlebars: Handlebars<'static>,
}

impl ShaderTemplate {
    pub fn new() -> ShaderTemplate {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_templates_directory("twgsl", "shader")
            .unwrap();
        handlebars.register_escape_fn(handlebars::no_escape);
        ShaderTemplate { handlebars }
    }

    pub fn get_shader(&self, shader_name: &str, data: &impl Serialize) -> String {
        self.handlebars.render(shader_name, data).unwrap()
    }
}
