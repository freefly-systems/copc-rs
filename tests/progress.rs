//! `write_with_progress` must report monotonic, non-decreasing progress that
//! reaches 100% across both the ingest and serialize phases, and must produce a
//! COPC with every input point preserved (the progress hook is instrumentation
//! only — it must not change the output).
use copc_rs::{BoundsSelection, CopcReader, CopcWriter, LodSelection};
use las::point::Format;
use las::{Builder, Point, Transform, Vector, Version, Vlr};

const EXTENT: f64 = 200.0;

fn test_header() -> las::Header {
    let version = Version::new(1, 4);
    let point_format = Format::new(6).unwrap();
    let raw = las::raw::Header {
        version,
        header_size: version.header_size(),
        offset_to_point_data: u32::from(version.header_size()),
        global_encoding: 16,
        point_data_record_format: point_format.to_u8().unwrap(),
        point_data_record_length: point_format.len(),
        x_scale_factor: 0.001,
        y_scale_factor: 0.001,
        z_scale_factor: 0.001,
        min_x: 0.0,
        max_x: EXTENT,
        min_y: 0.0,
        max_y: EXTENT,
        min_z: 0.0,
        max_z: 1.0,
        ..Default::default()
    };
    let mut builder = Builder::new(raw).unwrap();
    builder.transforms = Vector {
        x: Transform { scale: 0.001, offset: 0.0 },
        y: Transform { scale: 0.001, offset: 0.0 },
        z: Transform { scale: 0.001, offset: 0.0 },
    };
    builder.vlrs.push(Vlr {
        user_id: "LASF_Projection".to_string(),
        record_id: 2112,
        description: "WKT CRS".to_string(),
        data: b"LOCAL_CS[\"copc-rs test\"]".to_vec(),
    });
    builder.into_header().unwrap()
}

fn sweep_points(width: usize, height: usize) -> Vec<Point> {
    let step_x = EXTENT / width as f64;
    let step_y = EXTENT / height as f64;
    let mut pts = Vec::with_capacity(width * height);
    for x in 0..width {
        for y in 0..height {
            pts.push(Point {
                x: (x as f64 + 0.5) * step_x,
                y: (y as f64 + 0.5) * step_y,
                z: 0.5,
                gps_time: Some(pts.len() as f64),
                return_number: 1,
                number_of_returns: 1,
                ..Default::default()
            });
        }
    }
    pts
}

#[test]
fn progress_is_monotonic_completes_and_preserves_points() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("progress.copc.laz");
    // 40k points with a small max node size forces the voxel strategy and many
    // chunks, so BOTH the ingest sampling (every 4096 pts) and the per-chunk
    // serialize reporting are exercised.
    let points = sweep_points(200, 200);
    let n = points.len() as i32;

    let mut writer = CopcWriter::from_path(&path, test_header(), 0, 1000).unwrap();

    let mut samples: Vec<(u64, u64)> = Vec::new();
    writer
        .write_with_progress(points.clone(), n, |done, total| samples.push((done, total)))
        .unwrap();

    assert!(!samples.is_empty(), "no progress was reported");
    let total = samples[0].1;
    assert_eq!(total, 2 * points.len() as u64, "total should be 2 * num_points");
    assert!(
        samples.iter().all(|&(_, t)| t == total),
        "total changed mid-stream: {samples:?}"
    );
    assert!(
        samples.windows(2).all(|w| w[1].0 >= w[0].0),
        "progress went backwards: {samples:?}"
    );
    assert_eq!(samples.last().unwrap().0, total, "did not reach 100%");
    // crossed the phase boundary: at least one ingest sample (< base) and the
    // serialize tail (>= base) both present.
    let base = points.len() as u64;
    assert!(samples.iter().any(|&(d, _)| d < base), "no ingest-phase progress");
    assert!(samples.iter().any(|&(d, _)| d >= base), "no serialize-phase progress");

    // Instrumentation must not change the output: every input point survives.
    let mut reader = CopcReader::from_path(&path).unwrap();
    let count = reader
        .points(LodSelection::All, BoundsSelection::All)
        .unwrap()
        .count();
    assert_eq!(count, points.len(), "point count changed vs input");
}
