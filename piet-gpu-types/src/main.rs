fn main() {
    let mod_name = std::env::args()
        .skip(1)
        .next()
        .expect("provide a module name");
    match mod_name.as_str() {
        "scene" => print!("{}", piet_gpu_types::scene::gen_gpu_scene()),
        "ptcl" => print!("{}", piet_gpu_types::ptcl::gen_gpu_ptcl()),
        "test" => print!("{}", piet_gpu_types::test::gen_gpu_test()),
        _ => println!("Oops, unknown module name"),
    }
}
