use chrono::prelude::*;
use geo::point;
use geo::{prelude::*, Point};
use geojson::GeoJson;
use geojson::Geometry;
use geojson::Value;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use serde_json::value::Value as JsonValue;

#[derive(Debug, Clone, PartialEq)]
struct AcState {
    pub pos: Point<f64>,
    pub alt: f64,
    pub gs: f64,
    pub ts: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
struct Runway {
    pub airport_ident: String,
    pub pos: Point<f64>,
    pub heading: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Rejection {
    pub aircraft_hex: String,
    pub reject_state: AcState,
    pub runway: Runway,
}

impl Rejection {
    pub fn adsbx_url(&self) -> String {
        let ts = self.reject_state.ts;
        let start_time = ts.checked_sub_signed(chrono::Duration::minutes(5)).unwrap();
        let end_time = ts.checked_add_signed(chrono::Duration::minutes(2)).unwrap();
        let start_time_str = start_time.format("%H:%M").to_string();
        let end_time_str = end_time.format("%H:%M").to_string();
        format!(
            "https://globe.adsbexchange.com/?icao={}&zoom=7.0&showTrace=2022-01-10&startTime={}&endTime={}&trackLabels",
            self.aircraft_hex,
            start_time_str,
            end_time_str
        )
    }
}

fn process_geojson(gj: &GeoJson) -> HashMap<String, Vec<AcState>> {
    let mut aircraft = HashMap::new();
    match *gj {
        GeoJson::FeatureCollection(ref ctn) => {
            for feature in &ctn.features {
                if let Some(ref geom) = feature.geometry {
                    let pos = point_coords(geom);
                    let p = feature.properties.as_ref().unwrap();
                    let icao = p["hex"].as_str().unwrap();
                    // if icao != "a081ca" {
                    //     continue;
                    // }
                    if !p.contains_key("alt_baro") || p["alt_baro"].is_null() {
                        continue;
                    }
                    let alt = parse_alt(&p["alt_baro"]);
                    if !p.contains_key("gs") || p["gs"].is_null() {
                        continue;
                    }
                    let gs = p["gs"].as_f64().unwrap();
                    if !aircraft.contains_key(icao) {
                        aircraft.insert(icao.to_string(), vec![]);
                    }
                    let ts_secs = p["ts"].as_i64().unwrap();
                    let ts = Utc.timestamp(ts_secs, 0);
                    aircraft
                        .get_mut(icao)
                        .unwrap()
                        .push(AcState { pos, alt, gs, ts });
                }
            }
        }
        _ => {
            panic!("Unexpected thingy");
        }
    }
    aircraft
}

fn parse_alt(alt_str: &serde_json::value::Value) -> f64 {
    match alt_str {
        // JsonValue::Number(_) => alt_str.as_f64().unwrap(),
        JsonValue::Number(_) => 500.0,
        JsonValue::String(_) => 0.0,
        _ => panic!("Unexpected thingy"),
    }
}

/// Process GeoJSON geometries
fn point_coords(geom: &Geometry) -> Point<f64> {
    match &geom.value {
        Value::Point(pos) => point!(x: pos[0], y: pos[1]),
        _ => panic!("Matched some other geometry"),
    }
}

const NUM_ACCELS_THRESHOLD: u32 = 3;
const GS_THRESHOLD: f64 = 50.0;
const RUNWAY_MAX_DIST_M: f64 = 10000.0 / 3.28;

fn close_to_runway(runways: &[Runway], pos: Point<f64>) -> Option<&Runway> {
    for rwy in runways {
        let distance_m = rwy.pos.haversine_distance(&pos);
        if distance_m <= RUNWAY_MAX_DIST_M {
            return Some(rwy);
        }
    }
    None
}

pub fn heading_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

const FINAL_APPROACH_HEADING_DEG_EPSILON: f64 = 3.0;

fn close_to_and_in_line_with_runway(
    runways: &[Runway],
    pos: Point<f64>,
    heading: f64,
) -> Option<&Runway> {
    for rwy in runways {
        let distance_m = rwy.pos.haversine_distance(&pos);
        if distance_m <= RUNWAY_MAX_DIST_M
            && heading_diff(rwy.heading, heading) < FINAL_APPROACH_HEADING_DEG_EPSILON
        {
            return Some(rwy);
        }
    }
    None
}

fn process_runways(runway_str: &str) -> Vec<Runway> {
    let mut runways = vec![];
    let runway_csv = quick_csv::Csv::from_string(runway_str);
    for row in runway_csv {
        let r = row.unwrap();
        let mut cols = r.columns().unwrap();
        let airport_ident = cols.nth(2).unwrap().to_string();
        let lat_opt = cols.nth(6);
        let lon_opt = cols.next();
        let hdg_opt = cols.nth(1);
        if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
            if let (Ok(lat), Ok(lon)) = (lat.parse::<f64>(), lon.parse::<f64>()) {
                if let Some(hdg_opt) = hdg_opt {
                    if let Ok(hdg) = hdg_opt.parse::<f64>() {
                        runways.push(Runway {
                            airport_ident,
                            pos: point!(x: lon, y: lat),
                            heading: hdg,
                        });
                    }
                }
            }
        }
    }
    println!("Loaded {} runways", runways.len());
    let runways: Vec<Runway> = runways
        .iter()
        .cloned()
        .filter(|r| r.pos.x() < -113.0)
        .collect();
    println!("After geofiltering we have {} runways", runways.len());
    runways
}

fn detect_rejected_takeoff(
    runways: &[Runway],
    aircraft_hex: &str,
    history: &[AcState],
) -> Option<Rejection> {
    let mut num_accels = 0;
    let mut last_gs = 0.0;
    for s in history {
        if s.alt == 0.0 {
            if let Some(runway) = close_to_runway(runways, s.pos) {
                if s.gs > last_gs {
                    num_accels += 1;
                }
                if s.gs < last_gs {
                    if num_accels >= NUM_ACCELS_THRESHOLD && last_gs > GS_THRESHOLD {
                        println!("Rejected takeoff: {:?} {:?}", s, runway);
                        return Some(Rejection {
                            aircraft_hex: aircraft_hex.to_string(),
                            reject_state: s.clone(),
                            runway: runway.clone(),
                        });
                    }
                    num_accels = 0;
                }
                last_gs = s.gs;
            }
        } else {
            num_accels = 0;
            last_gs = 0.0;
        }
    }
    None
}

#[derive(Debug, PartialEq, Clone)]
pub struct GoAround {
    aircraft_hex: String,
    reject_state: AcState,
    runway: Runway,
}

impl GoAround {
    pub fn adsbx_url(&self) -> String {
        let ts = self.reject_state.ts;
        let start_time = ts.checked_sub_signed(chrono::Duration::minutes(0)).unwrap();
        let end_time = ts.checked_add_signed(chrono::Duration::minutes(2)).unwrap();
        let start_time_str = start_time.format("%H:%M").to_string();
        let end_time_str = end_time.format("%H:%M").to_string();
        format!(
            "https://globe.adsbexchange.com/?icao={}&zoom=15.0&showTrace=2022-01-10&startTime={}&endTime={}&trackLabels",
            self.aircraft_hex,
            start_time_str,
            end_time_str
        )
    }
}

fn heading(p1: Point<f64>, p2: Point<f64>) -> f64 {
    p1.bearing(p2)
}

fn on_final(
    runways: &[Runway],
    _aircraft_hex: &str,
    history: &[AcState],
    i: usize,
) -> Option<Runway> {
    let cur_state = &history[i];
    if cur_state.alt > 0.0 && cur_state.alt < 2000.0 && i + 4 < history.len() {
        let next_state = &history[i + 1];
        let heading = heading(cur_state.pos, next_state.pos);
        if let Some(runway) = close_to_and_in_line_with_runway(runways, cur_state.pos, heading) {
            if heading > runway.heading - 3.0
                && heading < runway.heading + 3.0
                && cur_state.alt - history[i + 4].alt > 200.0
            {
                return Some(runway.clone());
            }
        }
    }
    None
}

fn didnt_land(history: &[AcState], i: usize) -> bool {
    i + 200 < history.len()
        && !history[i..i + 200]
            .iter()
            .any(|s| s.alt == 0.0 || s.gs == 0.0)
}

fn detect_go_arounds(
    runways: &[Runway],
    aircraft_hex: &str,
    history: &[AcState],
) -> Option<GoAround> {
    for (i, state) in history.iter().enumerate() {
        if let Some(runway) = on_final(runways, aircraft_hex, history, i) {
            if didnt_land(history, i) {
                println!("Alt at t + 200: {}", history[i + 200].alt);
                return Some(GoAround {
                    aircraft_hex: aircraft_hex.to_string(),
                    reject_state: state.clone(),
                    runway,
                });
            }
        }
    }
    None
}

fn main() {
    let geojson_file = File::open("/Users/wiseman/data/ground-stop-20220110.geojson").unwrap();
    let mut geojson_buffered_reader = BufReader::new(geojson_file);
    let mut geojson_contents = String::new();
    geojson_buffered_reader
        .read_to_string(&mut geojson_contents)
        .unwrap();
    let geojson = geojson_contents.parse::<GeoJson>().unwrap();
    let aircraft = process_geojson(&geojson);
    println!("Read positions of {} aircraft", aircraft.len());

    let runway_data_file = File::open("/Users/wiseman/data/runways.csv").unwrap();
    let mut runway_buffered_reader = BufReader::new(runway_data_file);
    let mut runway_contents = String::new();
    runway_buffered_reader
        .read_to_string(&mut runway_contents)
        .unwrap();
    let runways = process_runways(&runway_contents);

    aircraft.par_iter().for_each(|(aircraft_hex, history)| {
        if let Some(rejection) = detect_rejected_takeoff(&runways, aircraft_hex, history) {
            println!("{}", rejection.adsbx_url());
        }
        if let Some(go_around) = detect_go_arounds(&runways, aircraft_hex, history) {
            println!("{} {:?}", go_around.adsbx_url(), go_around.runway);
        }
    });
}
