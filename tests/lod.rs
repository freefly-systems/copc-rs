use copc_rs::{BoundsSelection, CopcReader, CopcWriter, LodSelection};
use las::point::Format;
use las::{Bounds, Builder, Point, Transform, Vector, Version, Vlr};
use std::collections::HashSet;

const EXTENT: f64 = 200.0;
const ROOT_BIN_COUNT: usize = 4;

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
        x: Transform {
            scale: 0.001,
            offset: 0.0,
        },
        y: Transform {
            scale: 0.001,
            offset: 0.0,
        },
        z: Transform {
            scale: 0.001,
            offset: 0.0,
        },
    };
    builder.vlrs.push(Vlr {
        user_id: "LASF_Projection".to_string(),
        record_id: 2112,
        description: "WKT CRS".to_string(),
        data: b"LOCAL_CS[\"copc-rs test\"]".to_vec(),
    });
    builder.into_header().unwrap()
}

fn point_at(x: f64, y: f64, gps_time: f64) -> Point {
    Point {
        x,
        y,
        z: 0.5,
        gps_time: Some(gps_time),
        return_number: 1,
        number_of_returns: 1,
        ..Default::default()
    }
}

fn sweep_points(width: usize, height: usize) -> Vec<Point> {
    let step_x = EXTENT / width as f64;
    let step_y = EXTENT / height as f64;
    let mut points = Vec::with_capacity(width * height);
    for x in 0..width {
        for y in 0..height {
            points.push(point_at(
                (x as f64 + 0.5) * step_x,
                (y as f64 + 0.5) * step_y,
                points.len() as f64,
            ));
        }
    }
    points
}

fn occupied_xy_bins(points: &[Point], bins_per_axis: usize) -> usize {
    let mut occupied = HashSet::new();
    for point in points {
        let ix = ((point.x / EXTENT) * bins_per_axis as f64).floor() as usize;
        let iy = ((point.y / EXTENT) * bins_per_axis as f64).floor() as usize;
        occupied.insert((ix.min(bins_per_axis - 1), iy.min(bins_per_axis - 1)));
    }
    occupied.len()
}

#[test]
fn sweep_order_root_lod_spreads_across_the_extent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sweep.copc.laz");
    let points = sweep_points(40, 40);

    let mut writer = CopcWriter::from_path(&path, test_header(), 1, 64).unwrap();
    writer.write(points.clone(), points.len() as i32).unwrap();

    let mut reader = CopcReader::from_path(&path).unwrap();
    let all_count = reader
        .points(LodSelection::All, BoundsSelection::All)
        .unwrap()
        .count();
    assert_eq!(points.len(), all_count);

    let root_points: Vec<_> = reader
        .points(LodSelection::Level(0), BoundsSelection::All)
        .unwrap()
        .collect();
    let occupied = occupied_xy_bins(&root_points, ROOT_BIN_COUNT);
    assert_eq!(
        ROOT_BIN_COUNT * ROOT_BIN_COUNT,
        occupied,
        "root LOD should cover the whole XY extent, but {} root points occupied only {occupied}/{} coarse bins",
        root_points.len(),
        ROOT_BIN_COUNT * ROOT_BIN_COUNT
    );
}

#[test]
fn writer_options_can_force_spill_budget_without_breaking_losslessness() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("options.copc.laz");
    let spill_dir = dir.path().join("spill");
    let points = sweep_points(20, 20);
    let options = copc_rs::WriterOptions {
        memory_budget_bytes: 512,
        temp_dir: Some(spill_dir.clone()),
    };

    let mut writer =
        CopcWriter::from_path_with_options(&path, test_header(), 1, 64, options).unwrap();
    writer.write(points.clone(), points.len() as i32).unwrap();

    let mut reader = CopcReader::from_path(&path).unwrap();
    assert_eq!(
        points.len(),
        reader
            .points(LodSelection::All, BoundsSelection::All)
            .unwrap()
            .count()
    );
    assert!(
        !spill_dir.exists() || std::fs::read_dir(&spill_dir).unwrap().next().is_none(),
        "spill files should be cleaned up after successful close"
    );
}

#[test]
fn forced_spill_uses_configured_temp_dir() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid-spill.copc.laz");
    let spill_file = dir.path().join("not-a-directory");
    std::fs::write(&spill_file, b"not a directory").unwrap();
    let points = sweep_points(20, 20);
    let options = copc_rs::WriterOptions {
        memory_budget_bytes: 512,
        temp_dir: Some(spill_file),
    };

    let mut writer =
        CopcWriter::from_path_with_options(&path, test_header(), 1, 64, options).unwrap();
    assert!(
        writer.write(points.clone(), points.len() as i32).is_err(),
        "forced spill should try to use the configured temp directory"
    );
}

#[test]
fn cancellable_write_cleans_spill_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cancel.copc.laz");
    let spill_dir = dir.path().join("spill");
    let points = sweep_points(80, 80);
    let options = copc_rs::WriterOptions {
        memory_budget_bytes: 512,
        temp_dir: Some(spill_dir.clone()),
    };
    let mut writer =
        CopcWriter::from_path_with_options(&path, test_header(), 1, 64, options).unwrap();

    let mut polls = 0;
    let err = writer
        .write_cancellable(points, 80 * 80, || {
            polls += 1;
            polls > 1
        })
        .unwrap_err();

    assert!(matches!(err, copc_rs::Error::Cancelled));
    assert!(
        !spill_dir.exists() || std::fs::read_dir(&spill_dir).unwrap().next().is_none(),
        "spill files should be cleaned after cancellation"
    );
}

#[test]
fn colocated_points_terminate_and_remain_queryable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duplicates.copc.laz");
    let points: Vec<_> = (0..256).map(|i| point_at(100.0, 100.0, i as f64)).collect();

    let mut writer = CopcWriter::from_path(&path, test_header(), 1, 64).unwrap();
    writer.write(points.clone(), points.len() as i32).unwrap();

    let mut reader = CopcReader::from_path(&path).unwrap();
    let bounds = Bounds {
        min: Vector {
            x: 99.9,
            y: 99.9,
            z: 0.4,
        },
        max: Vector {
            x: 100.1,
            y: 100.1,
            z: 0.6,
        },
    };
    let queried_count = reader
        .points(LodSelection::All, BoundsSelection::Within(bounds))
        .unwrap()
        .count();
    assert_eq!(points.len(), queried_count);
}
