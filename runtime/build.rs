//! Finds the Vitis AI Runtime (headers and libraries of unilog, xir, vart) and compiles the C++
//! bridge.
//!
//! Environment variables:
//!     VITIS_AI_PREFIX   prefix the runtime was installed into (include/, lib/); ci/build_vart.sh
//!                       builds it. Default: /usr/local, then /usr.
//!
//! XRT is not linked: the DPU runner (libvart-dpu-runner) loads it when a model is loaded, so the
//! module also works on computers without XRT (to read models, for `opennve system`).

use std::env;
use std::path::{Path, PathBuf};

fn prefixes() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = env::var("VITIS_AI_PREFIX") {
        dirs.push(PathBuf::from(dir));
    }
    dirs.push(PathBuf::from("/usr/local"));
    dirs.push(PathBuf::from("/usr"));
    dirs
}

fn lib_dir(prefix: &Path) -> Option<PathBuf> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let triple = format!("{arch}-linux-gnu");
    [
        prefix.join("lib"),
        prefix.join("lib").join(triple),
        prefix.join("lib64"),
    ]
    .into_iter()
    .find(|dir| dir.join("libvart-runner.so").exists())
}

/// "5.0.0" from libvart-runner.so.5.0.0
fn version(lib_dir: &Path) -> String {
    std::fs::read_dir(lib_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_prefix("libvart-runner.so.")
                .filter(|version| version.matches('.').count() == 2)
                .map(str::to_owned)
        })
        .next()
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    println!("cargo:rerun-if-env-changed=VITIS_AI_PREFIX");
    println!("cargo:rerun-if-changed=cpp/shim.hpp");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=src/lib.rs");

    let (prefix, lib) = prefixes()
        .into_iter()
        .filter(|prefix| {
            prefix
                .join("include")
                .join("vart")
                .join("runner_ext.hpp")
                .is_file()
        })
        .find_map(|prefix| lib_dir(&prefix).map(|lib| (prefix, lib)))
        .unwrap_or_else(|| {
            panic!(
                "Vitis AI Runtime not found (include/vart/runner_ext.hpp, lib/libvart-runner.so). \
                 Build it with ci/build_vart.sh PREFIX and set VITIS_AI_PREFIX=PREFIX"
            )
        });
    let include = prefix.join("include");
    let version = version(&lib);

    println!("cargo:rustc-env=VITIS_AI_RUNTIME_VERSION={version}");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in ["vart-runner", "vart-util", "xir", "unilog", "glog"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:include={}", include.display());
    println!("cargo:lib_dir={}", lib.display()); // DEP_VART_LIB_DIR for the Python module

    cxx_build::bridge("src/lib.rs")
        .file("cpp/shim.cpp")
        .include(&include)
        .define("OPENNVE_VART_VERSION", format!("\"{version}\"").as_str())
        // the runtime's headers use these (as its own CMake build does)
        .define("GLOG_USE_GLOG_EXPORT", None)
        .define("UNILOG_USE_DLL", "1")
        .define("XIR_USE_DLL", "1")
        .define("VART_USE_DLL", "1")
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-ignored-qualifiers")
        .compile("vitis_runtime_shim");
}
