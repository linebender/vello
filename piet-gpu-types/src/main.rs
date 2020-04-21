fn main() {
    let mod_name = std::env::args()
        .skip(1)
        .next()
        .expect("provide a module name");
    match mod_name.as_str() {
        "scene" => print!("{}", piet_gpu_types::scene::gen_gpu_scene()),
        "tilegroup" => print!("{}", piet_gpu_types::tilegroup::gen_gpu_tilegroup()),
        "ptcl" => print!("{}", piet_gpu_types::ptcl::gen_gpu_ptcl()),
        _ => println!("Oops, unknown module name"),
    }
}
