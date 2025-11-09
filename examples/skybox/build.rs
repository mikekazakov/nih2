fn main() {
    pkg_config::probe_library("sdl3").unwrap();
}
