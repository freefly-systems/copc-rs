//! COPC file writer.

use crate::compressor::CopcCompressor;
use crate::copc::{CopcInfo, Entry, HierarchyPage, OctreeNode, VoxelKey};

use las::{Builder, Header};

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// enum for point data record format upgrades
enum UpgradePdrf {
    From1to6,  // upgrades (1=>6)
    From3to7,  // upgrades (3=>7)
    NoUpgrade, // 6, 7 and 8
}

impl UpgradePdrf {
    fn log_string(&self) -> &str {
        match self {
            UpgradePdrf::From1to6 => "Upgrading LAS PDRF from 1 to 6",
            UpgradePdrf::From3to7 => "Upgrading LAS PDRF from 3 to 7",
            UpgradePdrf::NoUpgrade => "COPC supports the given PDRF",
        }
    }
}

/// COPC file writer
pub struct CopcWriter<'a, W: 'a + Write + Seek> {
    is_closed: bool,
    start: u64,
    // point writer
    compressor: CopcCompressor<'a, W>,
    header: Header,
    // a page of the written entries
    hierarchy: HierarchyPage,
    min_node_size: i32,
    max_node_size: i32,
    copc_info: CopcInfo,
    // root node in octree, access point for the tree
    root_node: OctreeNode,
    chunk_store: ChunkStore,
    // Per-node voxel cells that already have a representative point.
    voxel_occupancy: HashMap<VoxelKey, HashSet<u64>>,
}

const DEFAULT_MAX_VOXEL_LEVEL: i32 = 18;
const MAX_VOXEL_LEVEL_CAP: i32 = 24;

#[derive(Clone, Debug)]
pub struct WriterOptions {
    pub memory_budget_bytes: u64,
    pub temp_dir: Option<PathBuf>,
}

impl Default for WriterOptions {
    fn default() -> Self {
        WriterOptions {
            memory_budget_bytes: 4 * 1024 * 1024 * 1024,
            temp_dir: None,
        }
    }
}

struct ChunkStore {
    memory_budget_bytes: u64,
    buffered_bytes: u64,
    temp_dir: PathBuf,
    chunks: HashMap<VoxelKey, ChunkBuffer>,
}

enum ChunkBuffer {
    InMemory {
        bytes: Vec<u8>,
        point_count: u32,
    },
    Spilled {
        path: PathBuf,
        point_count: u32,
        byte_count: u64,
    },
}

impl ChunkStore {
    fn new(options: &WriterOptions) -> crate::Result<Self> {
        Ok(ChunkStore {
            memory_budget_bytes: options.memory_budget_bytes,
            buffered_bytes: 0,
            temp_dir: options.temp_dir.clone().unwrap_or_else(default_spill_dir),
            chunks: HashMap::default(),
        })
    }

    fn append(&mut self, key: VoxelKey, bytes: Vec<u8>) -> crate::Result<()> {
        let byte_count = bytes.len() as u64;
        if matches!(self.chunks.get(&key), Some(ChunkBuffer::Spilled { .. })) {
            return self.append_to_spilled(key, bytes);
        }

        self.ensure_budget_for(bytes.len())?;
        if matches!(self.chunks.get(&key), Some(ChunkBuffer::Spilled { .. })) {
            return self.append_to_spilled(key, bytes);
        }

        if self.buffered_bytes + byte_count > self.memory_budget_bytes {
            if self.chunks.contains_key(&key) {
                self.spill_in_memory_chunk(&key)?;
                return self.append_to_spilled(key, bytes);
            }
            return self.write_new_spilled_chunk(key, bytes);
        }

        match self.chunks.get_mut(&key) {
            Some(ChunkBuffer::InMemory {
                bytes: existing,
                point_count,
            }) => {
                existing.extend_from_slice(&bytes);
                *point_count += 1;
            }
            Some(ChunkBuffer::Spilled { .. }) => unreachable!("spilling is added in Task 3"),
            None => {
                self.chunks.insert(
                    key,
                    ChunkBuffer::InMemory {
                        bytes,
                        point_count: 1,
                    },
                );
            }
        }
        self.buffered_bytes += byte_count;
        Ok(())
    }

    fn ensure_budget_for(&mut self, additional_bytes: usize) -> crate::Result<()> {
        let additional_bytes = additional_bytes as u64;
        while self.buffered_bytes + additional_bytes > self.memory_budget_bytes {
            if !self.spill_largest_in_memory_chunk()? {
                break;
            }
        }
        Ok(())
    }

    fn spill_largest_in_memory_chunk(&mut self) -> crate::Result<bool> {
        let key = self
            .chunks
            .iter()
            .filter_map(|(key, chunk)| match chunk {
                ChunkBuffer::InMemory { bytes, .. } => Some((key.clone(), bytes.len())),
                ChunkBuffer::Spilled { .. } => None,
            })
            .max_by_key(|(_, len)| *len)
            .map(|(key, _)| key);

        match key {
            Some(key) => {
                self.spill_in_memory_chunk(&key)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn spill_in_memory_chunk(&mut self, key: &VoxelKey) -> crate::Result<()> {
        let path = self.spill_path_for(key);
        fs::create_dir_all(&self.temp_dir)?;
        let Some(ChunkBuffer::InMemory { bytes, point_count }) = self.chunks.remove(key) else {
            return Ok(());
        };
        let byte_count = bytes.len() as u64;
        let mut file = File::create(&path)?;
        file.write_all(&bytes)?;
        self.buffered_bytes = self.buffered_bytes.saturating_sub(byte_count);
        self.chunks.insert(
            key.clone(),
            ChunkBuffer::Spilled {
                path,
                point_count,
                byte_count,
            },
        );
        Ok(())
    }

    fn write_new_spilled_chunk(&mut self, key: VoxelKey, bytes: Vec<u8>) -> crate::Result<()> {
        let byte_count = bytes.len() as u64;
        let path = self.spill_path_for(&key);
        fs::create_dir_all(&self.temp_dir)?;
        let mut file = File::create(&path)?;
        file.write_all(&bytes)?;
        self.chunks.insert(
            key,
            ChunkBuffer::Spilled {
                path,
                point_count: 1,
                byte_count,
            },
        );
        Ok(())
    }

    fn append_to_spilled(&mut self, key: VoxelKey, bytes: Vec<u8>) -> crate::Result<()> {
        let Some(ChunkBuffer::Spilled {
            path,
            point_count,
            byte_count,
        }) = self.chunks.get_mut(&key)
        else {
            return self.write_new_spilled_chunk(key, bytes);
        };

        let mut file = OpenOptions::new().append(true).open(path)?;
        file.write_all(&bytes)?;
        *point_count += 1;
        *byte_count += bytes.len() as u64;
        Ok(())
    }

    fn spill_path_for(&self, key: &VoxelKey) -> PathBuf {
        self.temp_dir.join(format!(
            "node-{}-{}-{}-{}.bin",
            key.level, key.x, key.y, key.z
        ))
    }

    fn remove(&mut self, key: &VoxelKey) -> Option<ChunkBuffer> {
        let chunk = self.chunks.remove(key)?;
        if let ChunkBuffer::InMemory { bytes, .. } = &chunk {
            self.buffered_bytes = self.buffered_bytes.saturating_sub(bytes.len() as u64);
        }
        Some(chunk)
    }

    fn remove_temp_dir_if_empty(&self) -> crate::Result<()> {
        match fs::remove_dir(&self.temp_dir) {
            Ok(()) => Ok(()),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::DirectoryNotEmpty
                        | std::io::ErrorKind::NotADirectory
                ) =>
            {
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn cleanup(&mut self) {
        for chunk in self.chunks.values() {
            if let ChunkBuffer::Spilled { path, .. } = chunk {
                let _ = fs::remove_file(path);
            }
        }
        let _ = fs::remove_dir(&self.temp_dir);
    }
}

fn default_spill_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("copc-rs-spill-{}-{nanos}", std::process::id()))
}

impl CopcWriter<'_, BufWriter<File>> {
    /// Creates a new COPC-writer for a path,
    /// creates a file at that path and wraps it in a BufWrite for you
    /// and passes it along to [new]
    ///
    /// see [new] for usage
    ///
    /// [new]: Self::new
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        header: Header,
        min_size: i32,
        max_size: i32,
    ) -> crate::Result<Self> {
        Self::from_path_with_options(path, header, min_size, max_size, WriterOptions::default())
    }

    pub fn from_path_with_options<P: AsRef<Path>>(
        path: P,
        header: Header,
        min_size: i32,
        max_size: i32,
        options: WriterOptions,
    ) -> crate::Result<Self> {
        let copc_ext = Path::new(match path.as_ref().file_stem() {
            Some(copc) => copc,
            None => return Err(crate::Error::WrongCopcExtension),
        })
        .extension();

        match (copc_ext, path.as_ref().extension()) {
            (Some(copc), Some(laz)) => match (&copc.to_str(), &laz.to_str()) {
                (Some(copc_str), Some(laz_str)) => {
                    if &copc_str.to_lowercase() != "copc" || &laz_str.to_lowercase() != "laz" {
                        return Err(crate::Error::WrongCopcExtension);
                    }
                }
                _ => return Err(crate::Error::WrongCopcExtension),
            },
            _ => return Err(crate::Error::WrongCopcExtension),
        }

        File::create(path)
            .map_err(crate::Error::from)
            .and_then(|file| {
                CopcWriter::new_with_options(
                    BufWriter::new(file),
                    header,
                    min_size,
                    max_size,
                    options,
                )
            })
    }
}

/// public API
impl<W: Write + Seek> CopcWriter<'_, W> {
    /// Create a COPC file writer for the write- and seekable `write`
    /// configured with the provided [las::Header]
    /// recommended to use [from_path] for writing to file
    ///
    /// The `bounds` field in the `header` is used as the bounds for the octree
    /// the bounds are checked for being normal
    ///
    /// `max_size` is the maximal number of [las::Point]s an octree node can hold
    /// any max_size < 1 sets the max_size to [crate::MAX_NODE_SIZE_DEFAULT]
    /// this is a soft limit
    ///
    /// `min_size` is the minimal number of [las::Point]s an octree node can hold
    /// any min_size < 1 sets the min_size to [crate::MIN_NODE_SIZE_DEFAULT]
    /// this is a hard limit
    ///
    /// `min_size` greater or equal to `max_size` after checking values < 1
    /// results in a [crate::Error::InvalidNodeSize] error
    ///
    ///
    /// This writer is strictly following the LAS 1.4 spec and the COPC spec
    /// which means that any provided header not compatible with those will lead
    /// to an Err
    /// That being said, LAS 1.2 headers and PDRFs 1 and 3 are accepted and upgraded to
    /// their matching LAS 1.4 versions
    /// GeoTiff CRS VLR's are parsed and written to WKT CRS VLR's
    /// A CRS VLR is __MANDATORY__ and without one
    ///
    /// [from_path]: Self::from_path
    pub fn new(write: W, header: Header, min_size: i32, max_size: i32) -> crate::Result<Self> {
        Self::new_with_options(write, header, min_size, max_size, WriterOptions::default())
    }

    pub fn new_with_options(
        mut write: W,
        header: Header,
        min_size: i32,
        max_size: i32,
        options: WriterOptions,
    ) -> crate::Result<Self> {
        let start = write.stream_position()?;

        let min_node_size = if min_size < 1 {
            crate::MIN_NODE_SIZE_DEFAULT
        } else {
            min_size
        };

        let max_node_size = if max_size < 1 {
            crate::MAX_NODE_SIZE_DEFAULT
        } else {
            max_size
        };

        if min_node_size >= max_node_size {
            return Err(crate::Error::InvalidNodeSize);
        }

        if header.version() != las::Version::new(1, 4) {
            log::log!(log::Level::Info, "Old Las version. Upgrading");
        }

        let mut has_wkt_vlr = false;

        // store the vlrs contained in the header for forwarding
        let mut forward_vlrs = Vec::with_capacity(header.vlrs().len());
        for vlr in header.vlrs() {
            match (vlr.user_id.to_lowercase().as_str(), vlr.record_id) {
                ("lasf_projection", 2112) => {
                    has_wkt_vlr = true;
                    forward_vlrs.push(vlr.clone());
                }
                // not forwarding these vlrs
                ("lasf_projection", 34735..=34737) => (), // geo-tiff crs
                ("copc", 1 | 1000) => (),
                ("laszip encoded", 22204) => (),
                ("lasf_spec", 100..355 | 65535) => (), // wave form packet descriptors
                // forwarding all other vlrs
                _ => forward_vlrs.push(vlr.clone()),
            }
        }

        // store the evlrs contained in the header for forwarding
        let mut forward_evlrs = Vec::with_capacity(header.evlrs().len());
        for evlr in header.evlrs() {
            match (evlr.user_id.to_lowercase().as_str(), evlr.record_id) {
                ("lasf_projection", 2112) => {
                    has_wkt_vlr = true;
                    forward_evlrs.push(evlr.clone());
                }
                // not forwarding these vlrs
                ("lasf_projection", 34735..=34737) => (), // geo-tiff crs
                ("copc", 1 | 1000) => (),                 // 1 should never be a evlr
                ("laszip encoded", 22204) => (),          // should never be a evlr
                ("lasf_spec", 100..355 | 65535) => (),    // waveform data packets
                // forwarding all other evlrs
                _ => forward_evlrs.push(evlr.clone()),
            }
        }

        // las version 1.4 says pdrf 6-10 must have a wkt crs
        // copc is only valid for las 1.4 and says only pdrf 6-8 is supported
        //
        // which means that any geotiff crs must be converted to a wkt crs
        //
        // could just use header.has_wkt_vlr(), but so many las files are wrongly written
        // so I don't trust it
        //
        // ignores any vertical crs that might stored in geotiff
        if !has_wkt_vlr {
            let epsg = las_crs::parse_las_crs(&header)?;
            let wkt_data = match crs_definitions::from_code(epsg.horizontal) {
                Some(wkt) => wkt,
                None => return Err(crate::Error::InvalidEPSGCode(epsg.horizontal)),
            }
            .wkt
            .as_bytes()
            .to_owned();

            let mut user_id = [0; 16];
            for (i, c) in "LASF_Projection".as_bytes().iter().enumerate() {
                user_id[i] = *c;
            }

            let crs_vlr = las::raw::Vlr {
                reserved: 0,
                user_id,
                record_id: 2112,
                record_length_after_header: las::raw::vlr::RecordLength::Vlr(wkt_data.len() as u16),
                description: [0; 32],
                data: wkt_data,
            };

            forward_vlrs.push(las::Vlr::new(crs_vlr));
        }

        // check bounds are normal
        let bounds = header.bounds();
        if !(bounds.max.x - bounds.min.x).is_normal()
            || !(bounds.max.y - bounds.min.y).is_normal()
            || !(bounds.max.z - bounds.min.z).is_normal()
        {
            return Err(crate::Error::InvalidBounds(bounds));
        }

        let mut raw_head = header.into_raw()?;

        // mask off the two leftmost bits corresponding to compression of pdrf
        let pdrf = raw_head.point_data_record_format & 0b00111111;
        let upgrade_pdrf = match pdrf {
            1 => {
                let upgrade = UpgradePdrf::From1to6;

                log::log!(log::Level::Info, "{}", upgrade.log_string());
                upgrade
            }
            3 => {
                let upgrade = UpgradePdrf::From3to7;

                log::log!(log::Level::Info, "{}", upgrade.log_string());
                upgrade
            }
            0 | 2 => {
                return Err(las::Error::InvalidPointFormat(las::point::Format::new(
                    raw_head.point_data_record_format,
                )?))?;
            }
            4..=5 | 9.. => {
                return Err(las::Error::InvalidPointFormat(las::point::Format::new(
                    raw_head.point_data_record_format,
                )?))?;
            }
            6..=8 => UpgradePdrf::NoUpgrade,
        };

        // adjust and clear some fields
        raw_head.version = las::Version::new(1, 4);
        raw_head.point_data_record_format += match upgrade_pdrf {
            UpgradePdrf::NoUpgrade => 0,
            UpgradePdrf::From1to6 => 5,
            UpgradePdrf::From3to7 => 4,
        };
        raw_head.point_data_record_format |= 0b11000000; // make sure the compress bits are set
        raw_head.point_data_record_length += match upgrade_pdrf {
            UpgradePdrf::NoUpgrade => 0,
            _ => 2,
        };
        raw_head.global_encoding |= 0b10000; // make sure wkt crs bit is set
        raw_head.number_of_point_records = 0;
        raw_head.number_of_points_by_return = [0; 5];
        raw_head.large_file = None;
        raw_head.evlr = None;
        raw_head.padding = vec![];

        let mut software_buffer = [0_u8; 32];
        for (i, byte) in format!("COPC-rs v{}", crate::VERSION).bytes().enumerate() {
            software_buffer[i] = byte;
        }
        raw_head.generating_software = software_buffer;

        // start building a real header from the raw header
        let mut builder = Builder::new(raw_head)?;
        // add a blank COPC-vlr as the first vlr
        builder.vlrs.push(CopcInfo::default().into_vlr()?);

        // create the laz vlr
        let point_format = builder.point_format;
        let mut laz_items = laz::laszip::LazItemRecordBuilder::new();
        laz_items.add_item(laz::LazItemType::Point14);
        if point_format.has_color {
            if point_format.has_nir {
                laz_items.add_item(laz::LazItemType::RGBNIR14);
            } else {
                laz_items.add_item(laz::LazItemType::RGB14);
            }
        }
        if point_format.extra_bytes > 0 {
            laz_items.add_item(laz::LazItemType::Byte14(point_format.extra_bytes));
        }

        let laz_vlr = laz::LazVlrBuilder::new(laz_items.build())
            .with_variable_chunk_size()
            .build();
        let mut cursor = Cursor::new(Vec::<u8>::new());
        laz_vlr.write_to(&mut cursor)?;
        let laz_vlr = las::Vlr {
            user_id: laz::LazVlr::USER_ID.to_owned(),
            record_id: laz::LazVlr::RECORD_ID,
            description: laz::LazVlr::DESCRIPTION.to_owned(),
            data: cursor.into_inner(),
        };
        builder.vlrs.push(laz_vlr);

        // add the forwarded vlrs
        builder.vlrs.extend(forward_vlrs);
        builder.evlrs.extend(forward_evlrs);
        // the EPT-hierarchy evlr is not yet added

        let header = builder.into_header()?;

        // write the header and vlrs
        // this is just to reserve the space
        header.write_to(&mut write)?;

        let center_point = las::Vector {
            x: (bounds.min.x + bounds.max.x) / 2.,
            y: (bounds.min.y + bounds.max.y) / 2.,
            z: (bounds.min.z + bounds.max.z) / 2.,
        };
        let halfsize = (center_point.x - bounds.min.x)
            .max((center_point.y - bounds.min.y).max(center_point.z - bounds.min.z));

        let mut root_node = OctreeNode::new();

        root_node.bounds = las::Bounds {
            min: las::Vector {
                x: center_point.x - halfsize,
                y: center_point.y - halfsize,
                z: center_point.z - halfsize,
            },
            max: las::Vector {
                x: center_point.x + halfsize,
                y: center_point.y + halfsize,
                z: center_point.z + halfsize,
            },
        };
        root_node.entry.key.level = 0;
        root_node.entry.offset = write.stream_position()?;

        let copc_info = CopcInfo {
            center: center_point,
            halfsize,
            spacing: 0.,
            root_hier_offset: 0,
            root_hier_size: 0,
            gpstime_minimum: f64::MAX,
            gpstime_maximum: f64::MIN,
        };

        let chunk_store = ChunkStore::new(&options)?;

        Ok(CopcWriter {
            is_closed: false,
            start,
            compressor: CopcCompressor::new(write, header.laz_vlr()?)?,
            header,
            hierarchy: HierarchyPage { entries: vec![] },
            min_node_size,
            max_node_size,
            copc_info,
            root_node,
            chunk_store,
            voxel_occupancy: HashMap::default(),
        })
    }

    /// Write anything that implements [IntoIterator]
    /// over [las::Point] to the COPC [Write]
    /// Only one iterator can be written so a call to [Self::write] closes the writer.
    ///
    /// `num_points` is the number of points in the iterator
    /// the number of points is used to choose between a single greedy chunk and
    /// deterministic voxel-grid LOD placement
    /// if `num_points` is < 1 a greedy filling strategy is used
    /// if `num_points` is not equal to the actual number of points in the
    /// iterator all points will still be written
    ///
    /// returns an `Err`([crate::Error::ClosedWriter]) if the writer has already been closed.
    ///
    /// If a point is outside the copc `bounds` or not matching the
    /// [las::point::Format] of the writer's header `Err` is returned
    /// [crate::PointAddError::PointAttributesDoNotMatch] take precedence over
    /// [crate::PointAddError::PointNotInBounds]
    /// All the points inside the bounds and matching the point format are written regardless
    ///
    /// All points which both match the point format and are inside the bounds are added
    ///
    /// Lastly [Self::close] is called. If closing fails an [crate::Error] is returned and
    /// the state of the [Write] is undefined
    ///
    /// If all points match the format, are inside the bounds and [Self::close] is successfull `Ok(())` is returned
    pub fn write<D: IntoIterator<Item = las::Point>>(
        &mut self,
        data: D,
        num_points: i32,
    ) -> crate::Result<()> {
        if self.is_closed {
            return Err(crate::Error::ClosedWriter);
        }

        let result = if num_points < self.max_node_size + self.min_node_size {
            // greedy filling strategy
            self.write_greedy(data)
        } else {
            // deterministic voxel-grid filling strategy
            self.write_voxel(data)
        };

        self.close()?;
        result
    }

    /// Like [`Self::write`] but polls `cancel` every 4096 points and aborts with
    /// [`crate::Error::Cancelled`] if it returns true. Closes the writer on success.
    pub fn write_cancellable<D, F>(
        &mut self,
        data: D,
        num_points: i32,
        mut cancel: F,
    ) -> crate::Result<()>
    where
        D: IntoIterator<Item = las::Point>,
        F: FnMut() -> bool,
    {
        if self.is_closed {
            return Err(crate::Error::ClosedWriter);
        }
        let greedy = num_points < self.max_node_size + self.min_node_size;
        let mut invalid = Ok(());
        for (i, p) in data.into_iter().enumerate() {
            if i % 4096 == 0 && cancel() {
                // copc-rs's Drop calls close().expect(), which panics on an
                // unfinished writer. Mark closed so the abort unwinds cleanly
                // (the partial file is the caller's to delete).
                self.chunk_store.cleanup();
                self.is_closed = true;
                return Err(crate::Error::Cancelled);
            }
            if !p.matches(self.header.point_format()) {
                invalid = Err(crate::Error::InvalidPoint(
                    crate::PointAddError::PointAttributesDoNotMatch(*self.header.point_format()),
                ));
                continue;
            }
            if !bounds_contains_point(&self.root_node.bounds, &p) {
                if invalid.is_ok() {
                    invalid = Err(crate::Error::InvalidPoint(
                        crate::PointAddError::PointNotInBounds,
                    ));
                }
                continue;
            }
            if greedy {
                self.add_point_greedy(p)?;
            } else {
                self.add_point_voxel(p)?;
            }
        }
        self.close()?;
        invalid
    }

    /// Like [`Self::write`] but reports build progress through `progress(done, total)`
    /// across **both** phases — point ingest *and* the close/serialize pass — so the
    /// bar never sprints to 100% and then stalls while `close()` compresses chunks.
    ///
    /// `total` is `2 * num_points`: every point is effectively touched twice — once
    /// decoded + voxel-placed on ingest, once LAZ-compressed on serialize — so the
    /// phases share the bar ~50/50, matching their real cost. The callback is
    /// throttled to whole-percent changes and, on ingest, only sampled every 4096
    /// points, so it adds no measurable per-point overhead. A terminal
    /// `progress(total, total)` is always emitted. Closes the writer like
    /// [`Self::write`].
    pub fn write_with_progress<D, F>(
        &mut self,
        data: D,
        num_points: i32,
        mut progress: F,
    ) -> crate::Result<()>
    where
        D: IntoIterator<Item = las::Point>,
        F: FnMut(u64, u64),
    {
        if self.is_closed {
            return Err(crate::Error::ClosedWriter);
        }
        let base = num_points.max(0) as u64;
        let total = base.saturating_mul(2).max(1);
        let greedy = num_points < self.max_node_size + self.min_node_size;
        let mut invalid = Ok(());
        let mut last_pct: i64 = -1;

        // Phase A — ingest: report points pulled from the source (0..=base), sampled
        // every 4096 points (a bitmask-cheap check, same cadence as write_cancellable).
        for (i, p) in data.into_iter().enumerate() {
            if i % 4096 == 0 {
                let done = i as u64;
                let pct = (done * 100 / total) as i64;
                if pct != last_pct {
                    last_pct = pct;
                    progress(done, total);
                }
            }
            if !p.matches(self.header.point_format()) {
                invalid = Err(crate::Error::InvalidPoint(
                    crate::PointAddError::PointAttributesDoNotMatch(*self.header.point_format()),
                ));
                continue;
            }
            if !bounds_contains_point(&self.root_node.bounds, &p) {
                if invalid.is_ok() {
                    invalid = Err(crate::Error::InvalidPoint(
                        crate::PointAddError::PointNotInBounds,
                    ));
                }
                continue;
            }
            if greedy {
                self.add_point_greedy(p)?;
            } else {
                self.add_point_voxel(p)?;
            }
        }

        // Phase B — serialize: each compressed chunk advances the bar past `base`.
        let mut compressed: u64 = 0;
        self.close_inner(|chunk_points| {
            compressed = compressed.saturating_add(chunk_points);
            let done = base.saturating_add(compressed).min(total);
            let pct = (done * 100 / total) as i64;
            if pct != last_pct {
                last_pct = pct;
                progress(done, total);
            }
        })?;

        progress(total, total); // guarantee a terminal 100%
        invalid
    }

    /// Whether this writer is closed or not
    pub fn is_closed(&self) -> bool {
        self.is_closed
    }

    /// number of points in the largest node
    pub fn max_node_size(&self) -> i32 {
        self.max_node_size
    }

    /// number of points in the smallest node
    pub fn min_node_size(&self) -> i32 {
        self.min_node_size
    }

    /// This writer's header, some fields are updated on closing of the writer
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// This writer's EPT Hierarchy
    pub fn hierarchy_entries(&self) -> &HierarchyPage {
        &self.hierarchy
    }

    /// This writer's COPC info
    pub fn copc_info(&self) -> &CopcInfo {
        &self.copc_info
    }
}

/// private functions
impl<W: Write + Seek> CopcWriter<'_, W> {
    fn voxel_grid_size(&self) -> i64 {
        ((self.max_node_size as f64).sqrt().round() as i64).max(2)
    }

    fn max_voxel_level(&self) -> i32 {
        let edge = 2.0 * self.copc_info.halfsize;
        let scale = self
            .header
            .transforms()
            .x
            .scale
            .abs()
            .min(self.header.transforms().y.scale.abs())
            .min(self.header.transforms().z.scale.abs());
        if edge.is_normal() && scale.is_normal() && scale.is_sign_positive() {
            (edge / scale).log2().ceil() as i32
        } else {
            DEFAULT_MAX_VOXEL_LEVEL
        }
        .clamp(0, MAX_VOXEL_LEVEL_CAP)
    }

    fn add_point_to_header(&mut self, point: &las::Point) {
        self.header.add_point(point);

        let gps_time = point.gps_time.unwrap_or(0.0);
        if gps_time < self.copc_info.gpstime_minimum {
            self.copc_info.gpstime_minimum = gps_time;
        }
        if gps_time > self.copc_info.gpstime_maximum {
            self.copc_info.gpstime_maximum = gps_time;
        }
    }

    fn write_point_to_open_chunk(&mut self, key: VoxelKey, point: las::Point) -> crate::Result<()> {
        let raw_point = point.into_raw(self.header.transforms())?;
        let mut bytes = vec![];
        raw_point.write_to(&mut bytes, self.header.point_format())?;
        self.chunk_store.append(key, bytes)
    }

    /// Voxel-grid strategy for writing points.
    fn write_voxel<D: IntoIterator<Item = las::Point>>(&mut self, data: D) -> crate::Result<()> {
        let mut invalid_points = Ok(());

        for p in data.into_iter() {
            if !p.matches(self.header.point_format()) {
                invalid_points = Err(crate::Error::InvalidPoint(
                    crate::PointAddError::PointAttributesDoNotMatch(*self.header.point_format()),
                ));
                continue;
            }
            if !bounds_contains_point(&self.root_node.bounds, &p) {
                if invalid_points.is_ok() {
                    invalid_points = Err(crate::Error::InvalidPoint(
                        crate::PointAddError::PointNotInBounds,
                    ));
                }
                continue;
            }

            self.add_point_voxel(p)?;
        }
        invalid_points
    }

    /// Greedy strategy for writing points
    fn write_greedy<D: IntoIterator<Item = las::Point>>(&mut self, data: D) -> crate::Result<()> {
        let mut invalid_points = Ok(());

        for p in data.into_iter() {
            if !p.matches(self.header.point_format()) {
                invalid_points = Err(crate::Error::InvalidPoint(
                    crate::PointAddError::PointAttributesDoNotMatch(*self.header.point_format()),
                ));
                continue;
            }
            if !bounds_contains_point(&self.root_node.bounds, &p) {
                if invalid_points.is_ok() {
                    invalid_points = Err(crate::Error::InvalidPoint(
                        crate::PointAddError::PointNotInBounds,
                    ));
                }
                continue;
            }

            self.add_point_greedy(p)?;
        }
        invalid_points
    }

    /// Close is called after the last point is written.
    fn close(&mut self) -> crate::Result<()> {
        self.close_inner(|_| {})
    }

    /// Like [`Self::close`] but invokes `on_chunk(point_count)` after each octree
    /// chunk is compressed and written out. The close pass LAZ-compresses every
    /// buffered chunk (and re-reads any spilled to disk), so it is a meaningful
    /// share of total build time — surfacing per-chunk completion lets a progress
    /// reporter span both the ingest and the serialize phases instead of stalling
    /// at "100%" while close() runs. The callback is the only addition; the write
    /// path is otherwise byte-for-byte identical to [`Self::close`].
    fn close_inner(&mut self, mut on_chunk: impl FnMut(u64)) -> crate::Result<()> {
        if self.is_closed {
            return Err(crate::Error::ClosedWriter);
        }
        if self.header.number_of_points() < 1 {
            return Err(crate::Error::EmptyCopcFile);
        }

        // write the unclosed chunks, order does not matter
        for (key, chunk) in self.chunk_store.chunks.drain() {
            let point_count = match chunk {
                ChunkBuffer::InMemory { bytes, .. } => {
                    if bytes.is_empty() {
                        continue;
                    }
                    let (chunk_table_entry, chunk_offset) =
                        self.compressor.compress_chunk(bytes)?;
                    self.hierarchy.entries.push(Entry {
                        key,
                        offset: chunk_offset,
                        byte_size: chunk_table_entry.byte_count as i32,
                        point_count: chunk_table_entry.point_count as i32,
                    });
                    chunk_table_entry.point_count as u64
                }
                ChunkBuffer::Spilled { path, .. } => {
                    let bytes = fs::read(&path)?;
                    let (chunk_table_entry, chunk_offset) =
                        self.compressor.compress_chunk(bytes)?;
                    fs::remove_file(&path)?;
                    self.hierarchy.entries.push(Entry {
                        key,
                        offset: chunk_offset,
                        byte_size: chunk_table_entry.byte_count as i32,
                        point_count: chunk_table_entry.point_count as i32,
                    });
                    chunk_table_entry.point_count as u64
                }
            };
            on_chunk(point_count);
        }
        self.chunk_store.remove_temp_dir_if_empty()?;

        self.compressor.done()?;

        let start_of_first_evlr = self.compressor.get_mut().stream_position()?;

        let raw_evlrs: Vec<las::Result<las::raw::Vlr>> = self
            .header
            .evlrs()
            .iter()
            .map(|evlr| evlr.clone().into_raw(true))
            .collect();

        // write copc-evlr
        self.hierarchy
            .clone()
            .into_evlr()?
            .into_raw(true)?
            .write_to(self.compressor.get_mut())?;
        // write the rest of the evlrs
        for raw_evlr in raw_evlrs {
            raw_evlr?.write_to(self.compressor.get_mut())?;
        }

        self.compressor
            .get_mut()
            .seek(SeekFrom::Start(self.start))?;
        self.header.clone().into_raw().and_then(|mut raw_header| {
            if let Some(mut e) = raw_header.evlr {
                e.start_of_first_evlr = start_of_first_evlr;
                e.number_of_evlrs += 1;
            } else {
                raw_header.evlr = Some(las::raw::header::Evlr {
                    start_of_first_evlr,
                    number_of_evlrs: 1,
                });
            }
            raw_header.write_to(self.compressor.get_mut())
        })?;

        // update the copc info vlr and write it
        self.copc_info.spacing = 2. * self.copc_info.halfsize / self.voxel_grid_size() as f64;
        self.copc_info.root_hier_offset = start_of_first_evlr + 60; // the header is 60bytes
        self.copc_info.root_hier_size = self.hierarchy.byte_size();

        self.copc_info
            .clone()
            .into_vlr()?
            .into_raw(false)?
            .write_to(self.compressor.get_mut())?;

        self.compressor
            .get_mut()
            .seek(SeekFrom::Start(self.start))?;

        self.is_closed = true;
        Ok(())
    }

    // find the first non-full octree-node that contains the point
    // and add it to the node, if the node now is full
    // add the node to the hierarchy page and write to file
    fn add_point_greedy(&mut self, point: las::Point) -> crate::Result<()> {
        self.add_point_to_header(&point);

        let mut node_key = None;
        let mut write_chunk = false;

        // starting from the root walk thorugh the octree
        // and find the correct node to add the point to
        let mut nodes_to_check = vec![&mut self.root_node];
        while let Some(node) = nodes_to_check.pop() {
            if !bounds_contains_point(&node.bounds, &point) {
                // the point does not belong to this subtree
                continue;
            }
            if node.is_full(self.max_node_size) {
                // the point belongs to the subtree, but this node is full
                // need to push the node's children to the nodes_to_check stack
                if node.children.is_empty() {
                    // the node does not have any children
                    // so lets add children to the node
                    // (split this node's bounds rather than re-deriving each
                    // child cube from the root — see add_point_voxel)
                    let center_x = (node.bounds.min.x + node.bounds.max.x) / 2.0;
                    let center_y = (node.bounds.min.y + node.bounds.max.y) / 2.0;
                    let center_z = (node.bounds.min.z + node.bounds.max.z) / 2.0;
                    let child_keys = node.entry.key.children();
                    for (dir, key) in child_keys.into_iter().enumerate() {
                        let child_bounds =
                            split_bounds(&node.bounds, center_x, center_y, center_z, dir as i32);
                        node.children.push(OctreeNode {
                            entry: Entry {
                                key,
                                offset: 0,
                                byte_size: 0,
                                point_count: 0,
                            },
                            bounds: child_bounds,
                            children: Vec::with_capacity(8),
                        })
                    }
                }
                // push the children to the stack
                for child in node.children.iter_mut() {
                    nodes_to_check.push(child);
                }
            } else {
                // we've found the first non-full node that contains the point
                node_key = Some(node.entry.key.clone());
                node.entry.point_count += 1;

                // check if the node now is full
                write_chunk = node.is_full(self.max_node_size);
                break;
            }
        }
        let Some(node_key) = node_key else {
            return Err(crate::Error::PointNotAddedToAnyNode);
        };

        self.write_point_to_open_chunk(node_key.clone(), point)?;

        if write_chunk {
            let chunk = self.chunk_store.remove(&node_key).unwrap();
            match chunk {
                ChunkBuffer::InMemory { bytes, .. } => {
                    let (chunk_table_entry, chunk_offset) =
                        self.compressor.compress_chunk(bytes)?;
                    self.hierarchy.entries.push(Entry {
                        key: node_key,
                        offset: chunk_offset,
                        byte_size: chunk_table_entry.byte_count as i32,
                        point_count: chunk_table_entry.point_count as i32,
                    });
                }
                ChunkBuffer::Spilled { path, .. } => {
                    let bytes = fs::read(&path)?;
                    let (chunk_table_entry, chunk_offset) =
                        self.compressor.compress_chunk(bytes)?;
                    fs::remove_file(&path)?;
                    self.hierarchy.entries.push(Entry {
                        key: node_key,
                        offset: chunk_offset,
                        byte_size: chunk_table_entry.byte_count as i32,
                        point_count: chunk_table_entry.point_count as i32,
                    });
                }
            }
        }
        Ok(())
    }

    fn add_point_voxel(&mut self, point: las::Point) -> crate::Result<()> {
        self.add_point_to_header(&point);

        let grid = self.voxel_grid_size();
        let max_level = self.max_voxel_level();
        let root_bounds = self.root_node.bounds;
        let mut key = VoxelKey {
            level: 0,
            x: 0,
            y: 0,
            z: 0,
        };
        let mut node_bounds = root_bounds;

        loop {
            let cell = voxel_cell(&node_bounds, &point, grid);
            let claimed = self
                .voxel_occupancy
                .entry(key.clone())
                .or_default()
                .insert(cell);

            if claimed || key.level >= max_level {
                return self.write_point_to_open_chunk(key, point);
            }

            let center_x = (node_bounds.min.x + node_bounds.max.x) / 2.0;
            let center_y = (node_bounds.min.y + node_bounds.max.y) / 2.0;
            let center_z = (node_bounds.min.z + node_bounds.max.z) / 2.0;
            let dir = (point.x >= center_x) as i32
                | (((point.y >= center_y) as i32) << 1)
                | (((point.z >= center_z) as i32) << 2);
            key = key.child(dir);
            // Split the parent bounds at the same centers used to pick `dir`
            // instead of re-deriving the child cube from the root
            // (`key.bounds` sizes every axis from the root's x edge, whose
            // rounding can disagree with the per-axis bounds by an ULP —
            // enough to put a point just outside the child it was assigned
            // to). Subdividing locally keeps containment exact by construction.
            node_bounds = split_bounds(&node_bounds, center_x, center_y, center_z, dir);
            debug_assert!(
                bounds_contains_point(&node_bounds, &point),
                "voxel descent must keep the point in the selected key bounds"
            );
        }
    }
}

impl<W: Write + Seek> Drop for CopcWriter<'_, W> {
    fn drop(&mut self) {
        if !self.is_closed {
            // can only happen if the writer is created but no points is written
            // or something goes wrong while writing
            if let Err(e) = self.close() {
                self.chunk_store.cleanup();
                panic!("Error when dropping the writer. No points written.: {e}");
            }
        }
    }
}

fn voxel_cell(bounds: &las::Bounds, point: &las::Point, grid: i64) -> u64 {
    let edge = bounds.max.x - bounds.min.x;
    let index = |value: f64, min: f64| -> i64 {
        (((value - min) / edge * grid as f64).floor() as i64).clamp(0, grid - 1)
    };
    let x = index(point.x, bounds.min.x);
    let y = index(point.y, bounds.min.y);
    let z = index(point.z, bounds.min.z);
    (x + y * grid + z * grid * grid) as u64
}

/// The `dir`-indexed octant of `b` split at the given center (bit 0 = upper x,
/// bit 1 = upper y, bit 2 = upper z) — mirrors `VoxelKey::child`. Unlike
/// `VoxelKey::bounds`, which sizes every axis from the root cube's x edge,
/// this subdivides the actual per-axis bounds, so a child face always equals
/// the parent's center/face bit-for-bit and octants partition the parent with
/// no ULP gaps.
#[inline]
fn split_bounds(b: &las::Bounds, cx: f64, cy: f64, cz: f64, dir: i32) -> las::Bounds {
    let upper = |bit: i32| dir >> bit & 1 == 1;
    las::Bounds {
        min: las::Vector {
            x: if upper(0) { cx } else { b.min.x },
            y: if upper(1) { cy } else { b.min.y },
            z: if upper(2) { cz } else { b.min.z },
        },
        max: las::Vector {
            x: if upper(0) { b.max.x } else { cx },
            y: if upper(1) { b.max.y } else { cy },
            z: if upper(2) { b.max.z } else { cz },
        },
    }
}

#[inline]
fn bounds_contains_point(b: &las::Bounds, p: &las::Point) -> bool {
    !(b.max.x < p.x
        || b.max.y < p.y
        || b.max.z < p.z
        || b.min.x > p.x
        || b.min.y > p.y
        || b.min.z > p.z)
}
