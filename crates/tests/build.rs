use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=VELLO_CI_NO_GPU");
    if env::var("VELLO_CI_NO_GPU").is_ok() {
        println!("cargo:rustc-cfg=skip_gpu_tests");
    }
}
