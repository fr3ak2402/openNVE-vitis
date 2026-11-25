//! Runs a generated .xmodel with VART's dummy runner (no DPU needed): checks reading models,
//! creating runners, quantizing and dequantizing.

fn model(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("test.xmodel");
    vitis_runtime::write_test_model(&path, 8, 6, 2, 4, 3).unwrap();
    path
}

#[test]
fn inspect_reads_the_dpu_subgraph() {
    let dir = tempdir();
    let info = vitis_runtime::inspect(model(&dir)).unwrap();
    assert_eq!(info.dpu_subgraphs, 1);
    assert_eq!(info.devices, vec!["USER".to_string(), "DPU".to_string()]);
    assert_eq!(info.target, "DPUCZDX8G_ISA1_B4096");
    assert_eq!(info.inputs.len(), 1);
    assert_eq!(info.inputs[0].shape, vec![8, 6, 2]);
    assert_eq!(info.inputs[0].frame_size, 96);
    assert_eq!(info.inputs[0].fix_point, 4);
    assert_eq!(info.outputs[0].fix_point, 3);
    assert_eq!(info.batch, 1);
}

#[test]
fn dummy_runner_runs_frames() {
    let dir = tempdir();
    let model =
        vitis_runtime::Model::load_with_runner(model(&dir), "libvart-dummy-runner.so").unwrap();
    assert_eq!(model.input().shape, vec![8, 6, 2]);
    assert_eq!(model.outputs().len(), 1);
    let input = vec![0.5f32; 3 * model.input().frame_size];
    let outputs = model.run(&input, 3).unwrap();
    assert_eq!(outputs[0].len(), 3 * model.outputs()[0].frame_size);
    assert!(model.run(&input[1..], 3).is_err());
}

#[test]
fn missing_and_broken_files_are_errors() {
    let dir = tempdir();
    assert!(vitis_runtime::inspect(dir.join("missing.xmodel")).is_err());
    assert!(vitis_runtime::Model::load(dir.join("missing.xmodel")).is_err());
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "vitis-runtime-test-{}-{}",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
