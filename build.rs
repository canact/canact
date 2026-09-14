fn main() {
    // Windows default stack is 1 MiB. rustls + reqwest + current-thread
    // Tokio on the CLI overflowed that on a 401 (cli_goldens).
    #[cfg(windows)]
    println!("cargo:rustc-link-arg=/STACK:8388608");
}
