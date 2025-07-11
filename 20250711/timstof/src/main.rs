// src/main.rs
mod utils;

use std::{collections::HashMap, env, error::Error, path::Path};
use polars::prelude::*;
use rayon::prelude::*;
use timsrust::{converters::ConvertableDomain, readers::FrameReader, MSLevel};

#[derive(Debug, Clone)]
pub struct TimsTOFData {
    pub rt_values_min: Vec<f64>,
    pub mobility_values: Vec<f64>,
    pub mz_values: Vec<f64>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<usize>,
    pub scan_indices: Vec<usize>,
}

impl TimsTOFData {
    pub fn new() -> Self {
        TimsTOFData {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }
    
    pub fn filter_by_mz_range(&self, min_mz: f64, max_mz: f64) -> Self {
        let mut filtered = TimsTOFData::new();
        
        for (i, &mz) in self.mz_values.iter().enumerate() {
            if mz >= min_mz && mz <= max_mz {
                filtered.rt_values_min.push(self.rt_values_min[i]);
                filtered.mobility_values.push(self.mobility_values[i]);
                filtered.mz_values.push(mz);
                filtered.intensity_values.push(self.intensity_values[i]);
                filtered.frame_indices.push(self.frame_indices[i]);
                filtered.scan_indices.push(self.scan_indices[i]);
            }
        }
        
        filtered
    }
    
    pub fn merge(data_list: Vec<TimsTOFData>) -> Self {
        let mut merged = TimsTOFData::new();
        
        for data in data_list {
            merged.rt_values_min.extend(data.rt_values_min);
            merged.mobility_values.extend(data.mobility_values);
            merged.mz_values.extend(data.mz_values);
            merged.intensity_values.extend(data.intensity_values);
            merged.frame_indices.extend(data.frame_indices);
            merged.scan_indices.extend(data.scan_indices);
        }
        
        merged
    }
    
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        let all_integers = self.mz_values.iter().all(|&mz| mz.fract() == 0.0);
        
        if all_integers {
            let mz_integers: Vec<i64> = self.mz_values.iter()
                .map(|&mz| mz as i64)
                .collect();
            
            let df = DataFrame::new(vec![
                Series::new("rt_values_min", &self.rt_values_min),
                Series::new("mobility_values", &self.mobility_values),
                Series::new("mz_values", mz_integers),
                Series::new("intensity_values", self.intensity_values.iter().map(|&v| v as f64).collect::<Vec<_>>()),
            ])?;
            Ok(df)
        } else {
            let df = DataFrame::new(vec![
                Series::new("rt_values_min", &self.rt_values_min),
                Series::new("mobility_values", &self.mobility_values),
                Series::new("mz_values", &self.mz_values),
                Series::new("intensity_values", self.intensity_values.iter().map(|&v| v as f64).collect::<Vec<_>>()),
            ])?;
            Ok(df)
        }
    }
}

/// What one peak looks like after we flatten a frame
#[derive(Debug, Clone)]
struct PeakRow {
    rt_values: f64,            // minutes
    mobility_values: f64,      // 1/K0
    mz_values: f64,            // m/z
    intensity_values: u32,
    frame_indices: u32,
    scan_indices: u32,
    precursor_indices: i32,    // 0 → MS1, otherwise MS2 window id
    quad_low_mz_values: f64,   // only set for MS2
    quad_high_mz_values: f64,  // only set for MS2
}

/// Flatten the whole .d folder into Vec<PeakRow>
fn explode_tdf(bruker_d: &Path) -> Result<Vec<PeakRow>, Box<dyn Error>> {
    let reader = FrameReader::new(bruker_d)?;
    let md     = timsrust::readers::MetadataReader::new(&bruker_d.join("analysis.tdf"))?;
    let mz_conv = md.mz_converter;
    let im_conv = md.im_converter;

    let rows: Vec<PeakRow> = (0..reader.len())
        .into_par_iter()
        .filter_map(|idx| reader.get(idx).ok())
        .flat_map(|frame| {
            let rt_min = frame.rt_in_seconds / 60.0;
            let mut local_rows = Vec::with_capacity(frame.tof_indices.len());

            match frame.ms_level {
                MSLevel::MS1 => {
                    for (pidx, (&tof, &inten)) in frame.tof_indices.iter()
                                                   .zip(frame.intensities.iter())
                                                   .enumerate() {
                        let scan   = utils::find_scan_for_index(pidx, &frame.scan_offsets) as u32;
                        let mz     = mz_conv.convert(tof as f64);
                        let im     = im_conv.convert(scan as f64);

                        local_rows.push(PeakRow {
                            rt_values        : rt_min,
                            mobility_values  : im,
                            mz_values        : mz,
                            intensity_values : inten,
                            frame_indices    : frame.index as u32,
                            scan_indices     : scan,
                            precursor_indices: 0,     // MS1 marker
                            quad_low_mz_values : 0.0,
                            quad_high_mz_values: 0.0,
                        });
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    // pre-compute every isolation window we have in that frame
                    let mut windows: Vec<(usize, usize, f64, f64)> = Vec::new(); // (start_scan,end_scan,low,high)
                    for i in 0..qs.isolation_mz.len() {
                        if i >= qs.isolation_width.len() { continue; }
                        let low  = qs.isolation_mz[i] - qs.isolation_width[i] / 2.0;
                        let high = qs.isolation_mz[i] + qs.isolation_width[i] / 2.0;
                        windows.push((qs.scan_starts[i], qs.scan_ends[i], low, high));
                    }

                    for (pidx, (&tof, &inten)) in frame.tof_indices.iter()
                                                   .zip(frame.intensities.iter())
                                                   .enumerate() {
                        let scan = utils::find_scan_for_index(pidx, &frame.scan_offsets);
                        // which quadrupole segment produced that scan?
                        if let Some((win_id, (low, high))) = windows.iter()
                            .enumerate()
                            .find_map(|(i,(s,e,l,h))|
                                      if scan >= *s && scan <= *e { Some((i,l,h)) } else { None })
                        {
                            let mz = mz_conv.convert(tof as f64);
                            let im = im_conv.convert(scan as f64);
                            local_rows.push(PeakRow {
                                rt_values        : rt_min,
                                mobility_values  : im,
                                mz_values        : mz,
                                intensity_values : inten,
                                frame_indices    : frame.index as u32,
                                scan_indices     : scan as u32,
                                precursor_indices: (win_id + 1) as i32, // keep 1-based id like alphatims
                                quad_low_mz_values : *low,
                                quad_high_mz_values: *high,
                            });
                        }
                    }
                }
                _ => {}
            }
            local_rows
        }).collect();

    Ok(rows)
}

/// Convert Vec<PeakRow> → polars DataFrame with identical column names to the Python code
fn rows_to_df(rows: Vec<PeakRow>) -> PolarsResult<DataFrame> {
    let len = rows.len();
    let mut col_rt      = Vec::with_capacity(len);
    let mut col_im      = Vec::with_capacity(len);
    let mut col_mz      = Vec::with_capacity(len);
    let mut col_int     = Vec::with_capacity(len);
    let mut col_frame   = Vec::with_capacity(len);
    let mut col_scan    = Vec::with_capacity(len);
    let mut col_precidx = Vec::with_capacity(len);
    let mut col_low     = Vec::with_capacity(len);
    let mut col_high    = Vec::with_capacity(len);

    for r in rows {
        col_rt.push(r.rt_values);
        col_im.push(r.mobility_values);
        col_mz.push(r.mz_values);
        col_int.push(r.intensity_values as f64);
        col_frame.push(r.frame_indices);
        col_scan.push(r.scan_indices);
        col_precidx.push(r.precursor_indices);
        col_low.push(r.quad_low_mz_values);
        col_high.push(r.quad_high_mz_values);
    }

    DataFrame::new(vec![
        Series::new("rt_values",           col_rt),
        Series::new("mobility_values",     col_im),
        Series::new("mz_values",           col_mz),
        Series::new("intensity_values",    col_int),
        Series::new("frame_indices",       col_frame),
        Series::new("scan_indices",        col_scan),
        Series::new("precursor_indices",   col_precidx),
        Series::new("quad_low_mz_values",  col_low),
        Series::new("quad_high_mz_values", col_high),
    ])
}

/// Same behaviour as the Python helper
pub struct FastChunkFinder {
    sorted_keys: Vec<(f64,f64)>,
    chunks     : HashMap<(f64,f64), DataFrame>,
}

impl FastChunkFinder {
    pub fn new(chunks: HashMap<(f64,f64), DataFrame>) -> Self {
        let mut keys: Vec<_> = chunks.keys().cloned().collect();
        keys.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
        FastChunkFinder { sorted_keys: keys, chunks }
    }

    pub fn find(&self, value: f64) -> Option<&DataFrame> {
        use bisect::Bisect;
        // We only have bisect in nightly; implement quick inline binary-search
        match self.sorted_keys.binary_search_by(|&(low,_)|
                    if value < low { std::cmp::Ordering::Greater } // we want last key <= value
                    else           { std::cmp::Ordering::Less    }) {
            Ok(_) | Err(0) => None,
            Err(idx) => {
                let key = self.sorted_keys[idx-1];
                if value >= key.0 && value <= key.1 {
                    self.chunks.get(&key)
                } else { None }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let bruker_path = env::args().nth(1)
        .expect("Usage: cargo run --release -- <bruker.d>");
    let bruker_d = Path::new(&bruker_path);
    println!("Reading {:?}", bruker_d);

    // 1. explode whole data set
    let rows   = explode_tdf(bruker_d)?;
    let mut df = rows_to_df(rows)?;

    // 2. sort by mz for faster binary search later
    df.set_sorted(vec![("mz_values".into(), true)])?;

    // 3. build MS1/MS2 slices like Python
    let df_ms1 = df.filter(&df.column("precursor_indices")?.i32()?.equal(0))?;
    let df_ms2 = df.filter(&df.column("precursor_indices")?.i32()?.not_equal(0))?;

    // 4. emulate pandas groupby on (low , high)
    let low   = df_ms2.column("quad_low_mz_values")?.f64()?;
    let high  = df_ms2.column("quad_high_mz_values")?.f64()?;
    let mut mapping: HashMap<(f64,f64), Vec<usize>> = HashMap::new();
    for (idx,(l,h)) in low.into_no_null_iter().zip(high.into_no_null_iter()).enumerate() {
        mapping.entry((l,h)).or_default().push(idx);
    }
    let mut chunks: HashMap<(f64,f64), DataFrame> = HashMap::new();
    for (key, idxs) in mapping {
        chunks.insert(key, df_ms2.take_iter( idxs.into_iter().map(|i| i as u32) )?);
    }

    // 5. construct fast finder
    let finder = FastChunkFinder::new(chunks);
    println!("Finder initialised – {} MS2 isolation windows detected", finder.sorted_keys.len());

    // 6. demo lookup
    let demo_mz = 500.0;
    if let Some(chunk) = finder.find(demo_mz) {
        println!("Demo – found {} rows for m/z={demo_mz}", chunk.height());
    } else {
        println!("Demo – m/z={demo_mz} not inside any MS2 isolation window");
    }

    // 7. optionally export the MS1/MS2 tables to feather/parquet/…
    // df_ms1.write_parquet("ms1.parquet", ParquetWriteOptions::default())?;

    Ok(())
}