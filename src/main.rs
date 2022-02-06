use chrono::prelude::*;
use geo::{point, Line};
use geo::{prelude::*, Point};
use geojson::GeoJson;
use geojson::Geometry;
use geojson::Value;
use rayon::prelude::*;
use std::time::Instant;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use serde_json::value::Value as JsonValue;

/// Aircraft state at a point in time.
#[derive(Debug, Clone, PartialEq)]
struct AcState {
    /// Position.
    pub pos: Point<f64>,
    /// Barometric altitude.
    pub alt: f64,
    /// Ground speed.
    pub gs: f64,
    pub ts: DateTime<Utc>,
}

/// Runway data.
///
/// Note that each physical runway is two logical runways (with opposing heading
/// and differen start positions), and each logical runway is a different Runway
/// struct.  E.g. KLAX runways 07L and 25R are the same physical runway.
#[derive(Debug, Clone, PartialEq)]
struct Runway {
    pub airport_ident: String,
    pub runway_ident: Option<String>,
    pub pos: Point<f64>,
    pub heading: f64,
    pub elevation_ft: f64,
    pub length_m: f64,
    pub width_m: f64,
    pub line: Line<f64>,
}

/// Represents a rejected takeoff.
#[derive(Debug, Clone, PartialEq)]
struct RejectedTakeoff {
    pub aircraft_hex: String,
    /// The state of the aircraft at the time we detected the rejection.
    pub reject_state: AcState,
    pub runway: Runway,
}

impl RejectedTakeoff {
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

/// Loads the GeoJSON file containing ADS-B data for all aircraft.
fn load_aircraft_data() -> HashMap<String, Vec<AcState>> {
    let start = Instant::now();
    let geojson_file = File::open("/Users/wiseman/data/ground-stop-20220110.geojson").unwrap();
    let mut geojson_buffered_reader = BufReader::new(geojson_file);
    // let geojson: GeoJson = serde_json::from_reader(&mut geojson_buffered_reader).unwrap();
    let mut geojson_contents = String::new();
    geojson_buffered_reader
        .read_to_string(&mut geojson_contents)
        .unwrap();
    let geojson = geojson_contents.parse::<GeoJson>().unwrap();
    println!(
        "Loaded aircraft geojson in {:.1} s",
        start.elapsed().as_millis() as f64 / 1000.0
    );
    let aircraft = process_geojson(&geojson);
    // Print some stats.
    println!(
        "Read {} positions of {} aircraft",
        aircraft.values().map(|v| v.len()).sum::<usize>(),
        aircraft.len()
    );
    let max_lon = aircraft
        .values()
        .map(|v| v.iter().map(|s| s.pos.x()))
        .flatten()
        .fold(f64::NAN, f64::max);
    println!("Max aircraft lon: {}", max_lon);
    let min_lon = aircraft
        .values()
        .map(|v| v.iter().map(|s| s.pos.x()))
        .flatten()
        .fold(f64::NAN, f64::min);
    println!("Min aircraft lon: {}", min_lon);
    let min_datetime = aircraft
    .values()
    .map(|v| v.iter().map(|s| s.ts))
    .flatten()
    .fold(chrono::MAX_DATETIME, |min, ts| min.min(ts));
    let max_datetime = aircraft
    .values()
    .map(|v| v.iter().map(|s| s.ts))
    .flatten()
    .fold(chrono::MIN_DATETIME, |max, ts| max.max(ts));
    println!("Min datetime: {}", min_datetime);
    println!("Max datetime: {}", max_datetime);
    aircraft
}

/// Parses the GeoJSON into aircraft states.
///
/// Returns a map from aircraft hex to a vector of states.
fn process_geojson(gj: &GeoJson) -> HashMap<String, Vec<AcState>> {
    let mut aircraft = HashMap::new();
    match *gj {
        GeoJson::FeatureCollection(ref ctn) => {
            for feature in &ctn.features {
                if let Some(ref geom) = feature.geometry {
                    let pos = point_coords(geom);
                    let p = feature.properties.as_ref().unwrap();
                    // If it doesn't have a hex property just die.
                    let icao = p["hex"].as_str().unwrap();
                    // if icao != "a081ca" {
                    //     continue;
                    // }
                    // If it doesn't have alt_baro or gs then skip it.
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
            panic!("Unexpected geojson thingy");
        }
    }
    aircraft
}

/// Parses the altitude from the "alt_baro" JSON property.
///
/// "ground" is converted to an altitude of 0.0.
fn parse_alt(alt_str: &serde_json::value::Value) -> f64 {
    match alt_str {
        // JsonValue::Number(_) => alt_str.as_f64().unwrap(),
        JsonValue::Number(x) => x.as_f64().unwrap(),
        JsonValue::String(_) => 0.0,
        _ => panic!("Unexpected thingy"),
    }
}

/// Parse an aircraft's GeoJSON position into a [geo::Point].
fn point_coords(geom: &Geometry) -> Point<f64> {
    match &geom.value {
        Value::Point(pos) => point!(x: pos[0], y: pos[1]),
        _ => panic!("Matched some other geometry"),
    }
}

/// Loads the CSV file containing runway data.
///
/// I'm using the runways.csv file available at <https://ourairports.com/data/>
fn load_runway_data() -> Vec<Runway> {
    let runway_data_file = File::open("/Users/wiseman/data/runways.csv").unwrap();
    let mut runway_buffered_reader = BufReader::new(runway_data_file);
    let mut runway_contents = String::new();
    runway_buffered_reader
        .read_to_string(&mut runway_contents)
        .unwrap();

    process_runways(&runway_contents)
}

/// Filter out any runways with longitude east of this since we're just looking
/// at the west coast of the U.S.
const RUNWAY_MAX_LONGITUDE: f64 = -107.0;

/// Parses the runways.csv data into a vector of [Runway]s.
fn process_runways(runway_str: &str) -> Vec<Runway> {
    let mut runways = vec![];
    let runway_csv = quick_csv::Csv::from_string(runway_str);
    // Each row has data for one physical runway, which we turn into two logical
    // runways.
    for row in runway_csv {
        let r = row.unwrap();
        let mut cols = r.columns().unwrap();
        let airport_ident = cols.nth(2).unwrap();
        let length_ft = cols.next();
        let width_ft = cols.next();
        let rwy_ident = cols.nth(3);
        let lat_opt = cols.next();
        let lon_opt = cols.next();
        let ele_opt = cols.next();
        let hdg_opt = cols.next();
        if let Some(runway) = runway_from_csv_values(
            airport_ident,
            rwy_ident,
            lat_opt,
            lon_opt,
            ele_opt,
            hdg_opt,
            length_ft,
            width_ft,
        ) {
            runways.push(runway);
        }
        let lat_opt = cols.nth(2);
        let lon_opt = cols.next();
        let ele_opt = cols.next();
        let hdg_opt = cols.next();
        if let Some(runway) = runway_from_csv_values(
            airport_ident,
            rwy_ident,
            lat_opt,
            lon_opt,
            ele_opt,
            hdg_opt,
            length_ft,
            width_ft,
        ) {
            runways.push(runway);
        }
    }
    println!("Loaded {} runways", runways.len());
    let runways: Vec<Runway> = runways
        .iter()
        .cloned()
        .filter(|r| r.pos.x() < RUNWAY_MAX_LONGITUDE)
        .collect();
    println!("After geofiltering we have {} runways", runways.len());
    runways
}

/// Parses runway.csv values into a [Runway], if possible.
#[allow(clippy::too_many_arguments)]
fn runway_from_csv_values(
    airport_ident: &str,
    runway_ident: Option<&str>,
    lat: Option<&str>,
    lon: Option<&str>,
    ele_ft: Option<&str>,
    hdg: Option<&str>,
    length_ft: Option<&str>,
    width_ft: Option<&str>,
) -> Option<Runway> {
    if lat.is_none() || lon.is_none() || hdg.is_none() {
        return None;
    }
    if let (Some(lat), Some(lon), Some(elevation), Some(heading), Some(length), Some(width_ft)) = (
        lat.and_then(|v| v.parse::<f64>().ok()),
        lon.and_then(|v| v.parse::<f64>().ok()),
        ele_ft.and_then(|v| v.parse::<f64>().ok()),
        hdg.and_then(|v| v.parse::<f64>().ok()),
        length_ft.and_then(|v| v.parse::<f64>().ok()),
        width_ft.and_then(|v| v.parse::<f64>().ok()),
    ) {
        // Build a geo::Line from the start of the runway to the end so we can
        // easily check the distance from any point to the runway.
        let start = point!(x: lon, y: lat);
        let end = start.haversine_destination(heading, length / 3.28);
        Some(Runway {
            airport_ident: airport_ident.to_string(),
            runway_ident: runway_ident.map(|v| v.to_string()),
            pos: point!(x: lon, y: lat),
            elevation_ft: elevation,
            heading,
            length_m: length / 3.28,
            width_m: width_ft / 3.28,
            line: Line::new(start, end),
        })
    } else {
        None
    }
}

const NUM_ACCELS_THRESHOLD: u32 = 3;
const GS_THRESHOLD: f64 = 50.0;

/// Returns the [Runway] a point is on, if any.
fn on_runway(runways: &[Runway], pos: Point<f64>) -> Option<&Runway> {
    for rwy in runways {
        let closest = rwy.line.closest_point(&pos);
        let distance = match closest {
            geo::Closest::Intersection(p) => pos.haversine_distance(&p),
            geo::Closest::SinglePoint(p) => pos.haversine_distance(&p),
            geo::Closest::Indeterminate => panic!("Indeterminate closest point"),
        };
        if distance < rwy.width_m / 2.0 {
            // println!("WOO\n{:?}\n{:?}\n{}" , rwy, pos, distance);
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

/// Determines the runway an aircraft is approaching, if any.
fn close_to_and_in_line_with_runway(
    runways: &[Runway],
    pos: Point<f64>,
    heading: f64,
) -> Option<&Runway> {
    for rwy in runways {
        let distance_m = rwy.pos.haversine_distance(&pos);
        // Calculate the bearing from aircraft to runway.
        let bearing_to_runway = pos.bearing(rwy.pos);
        // heading_diff is 0.0 if the aircraft is moving perfectly parallel with
        // the runway.
        let theta = deg2rad(heading_diff(bearing_to_runway, rwy.heading));
        // We use cos(theta) to scale the max allowable distance from the
        // threshold of the runway to the aircraft.
        if distance_m <= 3000.0 * theta.cos()
            && heading_diff(rwy.heading, heading) < FINAL_APPROACH_HEADING_DEG_EPSILON
        {
            return Some(rwy);
        }
    }
    None
}

fn deg2rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

/// Checks a single aircraft's history for a rejected takeoff.
fn detect_rejected_takeoff(
    runways: &[Runway],
    aircraft_hex: &str,
    history: &[AcState],
) -> Option<RejectedTakeoff> {
    let mut num_accels = 0;
    let mut last_gs = 0.0;
    for s in history {
        if s.alt == 0.0 {
            if let Some(runway) = on_runway(runways, s.pos) {
                if s.gs > last_gs {
                    num_accels += 1;
                }
                if s.gs < last_gs {
                    if num_accels >= NUM_ACCELS_THRESHOLD && last_gs > GS_THRESHOLD {
                        return Some(RejectedTakeoff {
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

/// Represents an aborted landing.
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
        let end_time = ts.checked_add_signed(chrono::Duration::minutes(5)).unwrap();
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

/// Checks whether an aircraft is considered to be on final approach at a
/// specific datapoint.
///
/// We'll say an aircraft is on final if
///
/// 1. It's between 0 and 2000 feet altitude.
/// 2. It's close to and in-line with a runway.
/// 3. If we look ahead 4 data points it's at least 200 feet lower than it is
///    now.
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

/// Checks whether an aircraft that is on final is considered to have _not
/// landed_ for at least the next 5 minutes.
///
/// We'll say we know an aircraft that is already on final didn't land if both
/// of these conditions are true:
///
/// 1. 5 minutes from now it's at a higher altitude than it is now.
/// 2. At no time between now and 5 minutes from now was it either low enough to
///    be on the runway or slow enough to be taxiiing.
fn did_not_land(history: &[AcState], i: usize, runway: &Runway) -> bool {
    let cur_time = history[i].ts;
    let future_time = cur_time
        .checked_add_signed(chrono::Duration::minutes(5))
        .unwrap();
    let intervening_states = history[i + 1..]
        .iter()
        .filter(|s| s.ts <= future_time)
        .collect::<Vec<_>>();
    let future_state = history[i + 1..].iter().find(|s| s.ts >= future_time);
    if let Some(future_state) = future_state {
        future_state.alt > history[i].alt + 500.0
            && !intervening_states
                .iter()
                .any(|s| s.alt < runway.elevation_ft + 20.0 || s.gs <= 50.0)
    } else {
        false
    }
}

/// Checks whether a single aircraft aborted a landing at any point in its
/// history.
///
/// We'll consider it a possible go-around if the aircraft was on final and then
/// didn't land.
fn detect_go_arounds(
    runways: &[Runway],
    aircraft_hex: &str,
    history: &[AcState],
) -> Option<GoAround> {
    for (i, state) in history.iter().enumerate() {
        if let Some(runway) = on_final(runways, aircraft_hex, history, i) {
            if did_not_land(history, i, &runway) {
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
    let aircraft = load_aircraft_data();
    let runways = load_runway_data();
    aircraft.par_iter().for_each(|(aircraft_hex, history)| {
        if let Some(rejection) = detect_rejected_takeoff(&runways, aircraft_hex, history) {
            println!("==== Rejected takeoff    {}", rejection.reject_state.ts);
            println!(
                "  {} {} {}",
                &rejection.aircraft_hex,
                &rejection.runway.airport_ident,
                &rejection
                    .runway
                    .runway_ident
                    .clone()
                    .unwrap_or_else(|| "".to_string())
            );
            println!("  {}\n", &rejection.adsbx_url());
        }
        if let Some(go_around) = detect_go_arounds(&runways, aircraft_hex, history) {
            println!("==== Go around           {}", go_around.reject_state.ts);
            println!(
                "  {} {} {}",
                &go_around.aircraft_hex,
                &go_around.runway.airport_ident,
                &go_around
                    .runway
                    .runway_ident
                    .clone()
                    .unwrap_or_else(|| "".to_string())
            );
            println!("  {}\n", &go_around.adsbx_url());
        }
    });
}
