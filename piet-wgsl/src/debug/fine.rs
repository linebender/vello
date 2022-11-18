#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Fill {
    pub tile: u32,
    pub backdrop: i32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Stroke {
    pub tile: u32,
    pub half_width: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Color {
    abgr: [u8; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct LinGrad {
    pub index: u32,
    pub line_x: f32,
    pub line_y: f32,
    pub line_c: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct RadGrad {
    pub index: u32,
    pub matrix: [f32; 4],
    pub xlat: [f32; 2],
    pub c1: [f32; 2],
    pub ra: f32,
    pub roff: f32,
}

#[derive(Copy, Clone, Debug)]
pub enum Command {
    Fill(Fill),
    Stroke(Stroke),
    Solid,
    Color(Color),
    LinGrad(LinGrad),
    RadGrad(RadGrad),
    BeginClip,
    EndClip(u32),
    End,
}

const PTCL_INITIAL_ALLOC: usize = 64;

#[derive(Debug)]
pub struct CommandList {
    pub tiles: Vec<(u32, u32, Vec<Command>)>,
}

impl CommandList {
    pub fn parse(width: usize, height: usize, ptcl: &[u8]) -> Self {
        let mut tiles = vec![];
        let width_tiles = width / 16;
        let height_tiles = height / 16;
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let tile_ix = y * width_tiles + x;
                let ix = tile_ix * PTCL_INITIAL_ALLOC;
                let commands = parse_commands(ptcl, ix);
                if !commands.is_empty() {
                    tiles.push((x as u32, y as u32, commands));
                }
            }
        }
        Self { tiles }
    }
}

fn parse_commands(ptcl: &[u8], mut ix: usize) -> Vec<Command> {
    let mut commands = vec![];
    let words: &[u32] = bytemuck::cast_slice(ptcl);
    while ix < words.len() {
        let tag = words[ix];
        ix += 1;
        match tag {
            0 => break,
            1 => {
                commands.push(Command::Fill(Fill {
                    tile: words[ix],
                    backdrop: words[ix + 1] as i32,
                }));
                ix += 2;
            }
            2 => {
                commands.push(Command::Stroke(Stroke {
                    tile: words[ix],
                    half_width: bytemuck::cast(words[ix + 1]),
                }));
                ix += 2;
            }
            3 => {
                commands.push(Command::Solid);
            }
            5 => {
                commands.push(Command::Color(Color {
                    abgr: bytemuck::cast(words[ix]),
                }));
                ix += 1;
            }
            6 => {
                commands.push(Command::LinGrad(LinGrad {
                    index: words[ix],
                    line_x: bytemuck::cast(words[ix + 1]),
                    line_y: bytemuck::cast(words[ix + 2]),
                    line_c: bytemuck::cast(words[ix + 3]),
                }));
                ix += 4;
            }
            7 => {
                let matrix = [
                    bytemuck::cast(words[ix + 1]),
                    bytemuck::cast(words[ix + 2]),
                    bytemuck::cast(words[ix + 3]),
                    bytemuck::cast(words[ix + 4]),
                ];
                let xlat = [bytemuck::cast(words[ix + 5]), bytemuck::cast(words[ix + 6])];
                let c1 = [bytemuck::cast(words[ix + 7]), bytemuck::cast(words[ix + 8])];
                commands.push(Command::RadGrad(RadGrad {
                    index: words[ix],
                    matrix,
                    xlat,
                    c1,
                    ra: bytemuck::cast(words[ix + 9]),
                    roff: bytemuck::cast(words[ix + 10]),
                }));
                ix += 11;
            }
            9 => {
                commands.push(Command::BeginClip);
            }
            10 => {
                commands.push(Command::EndClip(words[ix]));
                ix += 1;
            }
            11 => {
                ix = words[ix] as usize;
            }
            _ => {}
        }
    }
    commands
}
