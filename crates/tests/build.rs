use std::env;

fn main() {
    if env::var("VELLO_CI_NO_GPU").is_ok() {
        println!("cargo:rustc-cfg=skip_gpu_tests");
    }
}
