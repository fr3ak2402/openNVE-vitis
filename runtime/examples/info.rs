//! cargo run --example info [model.xmodel]: runtime version, DPU cores and the structure of a model.
fn main() {
    println!("Vitis AI Runtime {}", vitis_runtime::library_version());
    match vitis_runtime::dpu_cores() {
        Ok(cores) if cores.is_empty() => println!("no DPU loaded"),
        Ok(cores) => {
            for core in cores {
                println!(
                    "DPU {} fingerprint 0x{:x} batch {}",
                    core.name, core.fingerprint, core.batch
                );
            }
        }
        Err(error) => println!("DPU: {error}"),
    }
    if let Some(path) = std::env::args().nth(1) {
        match vitis_runtime::inspect(&path) {
            Ok(info) => println!("{path}: {info:#?}"),
            Err(error) => println!("{path}: {error}"),
        }
    }
}
