use std::collections::BTreeSet;

use parquet::basic::LogicalType;
use parquet::file::metadata::RowGroupMetaData;
use parquet::geospatial::statistics::GeospatialStatistics;
use parquet::schema::types::SchemaDescriptor;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeometryStats {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: Option<f64>,
    zmax: Option<f64>,
    mmin: Option<f64>,
    mmax: Option<f64>,
    types: BTreeSet<String>,
}

impl GeometryStats {
    pub(crate) fn from_json(value: &str) -> Option<Self> {
        let value: GeometryStatsJson = serde_json::from_str(value).ok()?;
        let types = value.types.into_iter().collect::<BTreeSet<_>>();
        let stats = Self {
            xmin: value.bbox.xmin,
            xmax: value.bbox.xmax,
            ymin: value.bbox.ymin,
            ymax: value.bbox.ymax,
            zmin: value.bbox.zmin,
            zmax: value.bbox.zmax,
            mmin: value.bbox.mmin,
            mmax: value.bbox.mmax,
            types,
        };
        stats.is_valid().then_some(stats)
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.xmin = self.xmin.min(other.xmin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymin = self.ymin.min(other.ymin);
        self.ymax = self.ymax.max(other.ymax);
        self.zmin = optional_min(self.zmin, other.zmin);
        self.zmax = optional_max(self.zmax, other.zmax);
        self.mmin = optional_min(self.mmin, other.mmin);
        self.mmax = optional_max(self.mmax, other.mmax);
        self.types.extend(other.types.iter().cloned());
    }

    pub(crate) fn to_json(&self) -> String {
        format!(
            "{{\"bbox\": {{\"xmin\": {:.6}, \"xmax\": {:.6}, \"ymin\": {:.6}, \"ymax\": {:.6}, \"zmin\": {}, \"zmax\": {}, \"mmin\": {}, \"mmax\": {}}}, \"types\": [{}]}}",
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            optional_json(self.zmin),
            optional_json(self.zmax),
            optional_json(self.mmin),
            optional_json(self.mmax),
            self.types
                .iter()
                .map(|value| format!("\"{value}\""))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    fn from_parquet(value: &GeospatialStatistics) -> Option<Self> {
        let bbox = value.bounding_box()?;
        let types = value
            .geospatial_types()?
            .iter()
            .copied()
            .map(geometry_type_name)
            .collect::<Option<BTreeSet<_>>>()?;
        let stats = Self {
            xmin: bbox.get_xmin(),
            xmax: bbox.get_xmax(),
            ymin: bbox.get_ymin(),
            ymax: bbox.get_ymax(),
            zmin: bbox.get_zmin(),
            zmax: bbox.get_zmax(),
            mmin: bbox.get_mmin(),
            mmax: bbox.get_mmax(),
            types,
        };
        stats.is_valid().then_some(stats)
    }

    fn is_valid(&self) -> bool {
        self.xmin.is_finite()
            && self.xmax.is_finite()
            && self.ymin.is_finite()
            && self.ymax.is_finite()
            && self.xmin <= self.xmax
            && self.ymin <= self.ymax
            && valid_optional_range(self.zmin, self.zmax)
            && valid_optional_range(self.mmin, self.mmax)
            && !self.types.is_empty()
            && self
                .types
                .iter()
                .all(|value| geometry_type_code(value).is_some())
    }
}

pub(crate) fn collect_geometry_stats(
    row_groups: &[RowGroupMetaData],
    schema: &SchemaDescriptor,
    leaf: usize,
) -> Option<String> {
    if !matches!(
        schema.column(leaf).logical_type_ref(),
        Some(LogicalType::Geometry(_))
    ) {
        return None;
    }

    let mut combined: Option<GeometryStats> = None;
    for row_group in row_groups {
        let chunk = row_group.column(leaf);
        let null_count = chunk.statistics()?.null_count_opt()?;
        let null_count = i64::try_from(null_count).ok()?;
        let value_count = chunk.num_values().checked_sub(null_count)?;
        if value_count == 0 {
            continue;
        }
        if value_count < 0 {
            return None;
        }
        let current = GeometryStats::from_parquet(chunk.geo_statistics()?)?;
        match &mut combined {
            Some(combined) => combined.merge(&current),
            None => combined = Some(current),
        }
    }
    combined.map(|stats| stats.to_json())
}

fn geometry_type_name(value: i32) -> Option<String> {
    let dimension = value / 1000;
    let base = value % 1000;
    let name = match base {
        1 => "point",
        2 => "linestring",
        3 => "polygon",
        4 => "multipoint",
        5 => "multilinestring",
        6 => "multipolygon",
        7 => "geometrycollection",
        _ => return None,
    };
    let suffix = match dimension {
        0 => "",
        1 => "_z",
        2 => "_m",
        3 => "_zm",
        _ => return None,
    };
    Some(format!("{name}{suffix}"))
}

fn geometry_type_code(value: &str) -> Option<i32> {
    for dimension in 0..=3 {
        for base in 1..=7 {
            let code = dimension * 1000 + base;
            if geometry_type_name(code).as_deref() == Some(value) {
                return Some(code);
            }
        }
    }
    None
}

fn optional_min(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}

fn optional_max(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    }
}

fn optional_json(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("{value:.6}"))
}

fn valid_optional_range(min: Option<f64>, max: Option<f64>) -> bool {
    match (min, max) {
        (None, None) => true,
        (Some(min), Some(max)) => min.is_finite() && max.is_finite() && min <= max,
        _ => false,
    }
}

#[derive(Deserialize)]
struct GeometryStatsJson {
    bbox: GeometryBboxJson,
    types: Vec<String>,
}

#[derive(Deserialize)]
struct GeometryBboxJson {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: Option<f64>,
    zmax: Option<f64>,
    mmin: Option<f64>,
    mmax: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_stats_merge_matches_ducklake_json() {
        let mut point = GeometryStats::from_json(
            r#"{"bbox": {"xmin": 1.000000, "xmax": 1.000000, "ymin": 2.000000, "ymax": 2.000000, "zmin": null, "zmax": null, "mmin": null, "mmax": null}, "types": ["point"]}"#,
        )
        .unwrap();
        let line = GeometryStats::from_json(
            r#"{"bbox": {"xmin": 5.000000, "xmax": 10.000000, "ymin": 5.000000, "ymax": 10.000000, "zmin": 5.000000, "zmax": 10.000000, "mmin": null, "mmax": null}, "types": ["linestring_z"]}"#,
        )
        .unwrap();
        point.merge(&line);
        assert_eq!(
            point.to_json(),
            r#"{"bbox": {"xmin": 1.000000, "xmax": 10.000000, "ymin": 2.000000, "ymax": 10.000000, "zmin": 5.000000, "zmax": 10.000000, "mmin": null, "mmax": null}, "types": ["linestring_z", "point"]}"#
        );
    }

    #[test]
    fn geometry_stats_reject_invalid_bounds_and_types() {
        assert_eq!(
            GeometryStats::from_json(
                r#"{"bbox": {"xmin": 2, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": null, "zmax": null, "mmin": null, "mmax": null}, "types": ["point"]}"#
            ),
            None
        );
        assert_eq!(
            GeometryStats::from_json(
                r#"{"bbox": {"xmin": 1, "xmax": 2, "ymin": 0, "ymax": 1, "zmin": null, "zmax": null, "mmin": null, "mmax": null}, "types": ["circle"]}"#
            ),
            None
        );
    }

    #[test]
    fn geometry_type_names_include_dimensions() {
        assert_eq!(geometry_type_name(1).as_deref(), Some("point"));
        assert_eq!(geometry_type_name(1002).as_deref(), Some("linestring_z"));
        assert_eq!(geometry_type_name(2001).as_deref(), Some("point_m"));
        assert_eq!(geometry_type_name(3001).as_deref(), Some("point_zm"));
        assert_eq!(geometry_type_name(8), None);
        assert_eq!(geometry_type_name(4001), None);
    }
}
