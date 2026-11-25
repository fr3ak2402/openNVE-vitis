//! The bundled Vitis AI Runtime lies next to the module in opennve_vitis/lib (copied there by
//! ci/repair_wheel.py): the module finds it through the rpath $ORIGIN/lib. For builds against a
//! runtime outside the system directories (development: VITIS_AI_PREFIX) that directory is added
//! too.
fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    if let Ok(dir) = std::env::var("DEP_VART_LIB_DIR") {
        if !dir.starts_with("/usr") && std::env::var("OPENNVE_VITIS_NO_RPATH").is_err() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
    println!("cargo:rerun-if-env-changed=OPENNVE_VITIS_NO_RPATH");
}
