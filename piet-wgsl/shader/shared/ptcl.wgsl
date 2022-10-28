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

// Tags for PTCL commands
let CMD_END = 0u;
let CMD_FILL = 1u;
let CMD_SOLID = 3u;
let CMD_COLOR = 5u;
let CMD_JUMP = 11u;

// The individual PTCL structs are written here, but read/write is by
// hand in the relevant shaders

struct CmdFill {
    tile: u32,
    backdrop: i32,
}

struct CmdJump {
    target: u32,
}

struct CmdColor {
    rgba_color: u32,
}
