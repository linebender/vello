fn main() {
    let mod_name = std::env::args()
        .skip(1)
        .next()
        .expect("provide a module name");
    match mod_name.as_str() {
        "scene" => print!("{}", piet_gpu_types::scene::gen_gpu_scene()),
        "state" => print!("{}", piet_gpu_types::state::gen_gpu_state()),
        "annotated" => print!("{}", piet_gpu_types::annotated::gen_gpu_annotated()),
        "pathseg" => print!("{}", piet_gpu_types::pathseg::gen_gpu_pathseg()),
        "bins" => print!("{}", piet_gpu_types::bins::gen_gpu_bins()),
        "tile" => print!("{}", piet_gpu_types::tile::gen_gpu_tile()),
        "tilegroup" => print!("{}", piet_gpu_types::tilegroup::gen_gpu_tilegroup()),
        "ptcl" => print!("{}", piet_gpu_types::ptcl::gen_gpu_ptcl()),
        "test" => print!("{}", piet_gpu_types::test::gen_gpu_test()),
        _ => println!("Oops, unknown module name"),
    }
}
