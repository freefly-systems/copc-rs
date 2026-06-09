//! Regression test for the voxel-descent cross-axis ULP gap.
//!
//! The descent picks a child octant by comparing the point against the parent
//! bounds' per-axis averaged centers, but used to re-derive the child cube
//! from the root via `VoxelKey::bounds`, which sizes EVERY axis from the root
//! cube's X edge ("every cell is a cube"). The root cube, however, is
//! constructed per-axis as `fl(center_axis ± halfsize)` — three different
//! centers, three different roundings — so its y/z extents can differ from
//! the x edge by a few ULPs. A point quantized bit-exactly onto an averaged
//! center plane then descends into a child whose re-derived face sits one ULP
//! past it, tripping
//! `debug_assert!("voxel descent must keep the point in the selected key bounds")`.
//!
//! The constants below are the exact bit patterns captured from a real scan
//! (`oldprodmission`, ECEF meters): the violation is on the z axis at level 1,
//! key (0,1,1). The header bounds equal the captured root cube, which is a
//! fixed point of the writer's cube derivation, so the writer reconstructs the
//! failing geometry exactly.
use copc_rs::{BoundsSelection, CopcReader, CopcWriter, LodSelection};
use las::point::Format;
use las::{Builder, Point, Transform, Vector, Version, Vlr};

const MIN_BITS: [u64; 3] = [0xc141739096f9db23, 0xc14bc02dbc28f5c3, 0x4151ebbd7bd70a3e];
const MAX_BITS: [u64; 3] = [0xc14172c92d810625, 0xc14bbf6652b020c5, 0x4151ec21309374bc];
// z is bit-identical to the root's averaged z center; the level-1 upper-z face
// re-derived from the x edge lands one ULP above it.
const TRAP_POINT_BITS: [u64; 3] = [0xc141733bb6666666, 0xc14bbfc39db22d0e, 0x4151ebef56353f7d];

const SCALE: f64 = 0.001;

fn header_for(min: [f64; 3], max: [f64; 3]) -> las::Header {
    // offsets keep (value - offset) / scale within i32, like any ECEF LAS file
    let offset = [
        ((min[0] + max[0]) / 2.0).round(),
        ((min[1] + max[1]) / 2.0).round(),
        ((min[2] + max[2]) / 2.0).round(),
    ];
    let version = Version::new(1, 4);
    let point_format = Format::new(6).unwrap();
    let raw = las::raw::Header {
        version,
        header_size: version.header_size(),
        offset_to_point_data: u32::from(version.header_size()),
        global_encoding: 16,
        point_data_record_format: point_format.to_u8().unwrap(),
        point_data_record_length: point_format.len(),
        x_scale_factor: SCALE,
        y_scale_factor: SCALE,
        z_scale_factor: SCALE,
        x_offset: offset[0],
        y_offset: offset[1],
        z_offset: offset[2],
        min_x: min[0],
        max_x: max[0],
        min_y: min[1],
        max_y: max[1],
        min_z: min[2],
        max_z: max[2],
        ..Default::default()
    };
    let mut builder = Builder::new(raw).unwrap();
    builder.transforms = Vector {
        x: Transform {
            scale: SCALE,
            offset: offset[0],
        },
        y: Transform {
            scale: SCALE,
            offset: offset[1],
        },
        z: Transform {
            scale: SCALE,
            offset: offset[2],
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

#[test]
fn point_on_averaged_center_plane_descends_without_panicking() {
    let min = MIN_BITS.map(f64::from_bits);
    let max = MAX_BITS.map(f64::from_bits);
    let trap = TRAP_POINT_BITS.map(f64::from_bits);

    // Two identical points in one root voxel cell: the first claims the cell,
    // the second must descend to level 1 across the mismatched z face.
    let points: Vec<Point> = (0..2)
        .map(|i| Point {
            x: trap[0],
            y: trap[1],
            z: trap[2],
            gps_time: Some(i as f64),
            return_number: 1,
            number_of_returns: 1,
            ..Default::default()
        })
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ulp-trap.copc.laz");
    let mut writer = CopcWriter::from_path(&path, header_for(min, max), 1, 64).unwrap();
    // An inflated advisory count selects the deterministic voxel strategy.
    writer.write(points, 1_000).unwrap();

    let mut reader = CopcReader::from_path(&path).unwrap();
    let count = reader
        .points(LodSelection::All, BoundsSelection::All)
        .unwrap()
        .count();
    assert_eq!(2, count);
}
