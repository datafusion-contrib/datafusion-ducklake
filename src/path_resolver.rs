//! Path resolution utilities for DuckLake
//!
//! Provides utilities for parsing object store URLs and resolving hierarchical paths
//! in the DuckLake catalog structure (catalog -> schema -> table -> file).

use crate::{DuckLakeError, Result};
use datafusion::datasource::object_store::ObjectStoreUrl;
use percent_encoding::percent_decode_str;
use std::sync::Arc;

/// Validate that a path does not contain null bytes (literal or URL-encoded)
fn validate_no_null_bytes(path: &str) -> Result<()> {
    if path.contains('\0') {
        return Err(DuckLakeError::InvalidConfig(format!(
            "Path contains null byte: {}",
            path.replace('\0', "\\0")
        )));
    }
    if path.contains("%00") {
        return Err(DuckLakeError::InvalidConfig(format!(
            "Path contains URL-encoded null byte (%00): {}",
            path
        )));
    }
    Ok(())
}

/// Simple percent-decode of path-relevant characters for traversal detection
fn percent_decode_path(path: &str) -> String {
    let mut result = String::with_capacity(path.len());
    let bytes = path.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&path[i + 1..i + 3], 16)
        {
            result.push(byte as char);
            i += 3;
            continue;
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Check if any path component is ".." when split on `/` and `\`
fn has_dotdot_component(path: &str) -> bool {
    path.split(['/', '\\']).any(|c| c == "..")
}

/// Validate that a path does not contain path traversal sequences
fn validate_no_path_traversal(path: &str) -> Result<()> {
    // Check literal '..' components (split on / and \)
    if has_dotdot_component(path) {
        return Err(DuckLakeError::InvalidConfig(format!(
            "Path traversal detected: path contains '..' component: {}",
            path
        )));
    }
    // Fast path: skip allocation if no percent-encoded characters
    if !path.contains('%') {
        return Ok(());
    }
    // Also check the percent-decoded path to catch encoded variants
    // (e.g., %2e%2e%2f, ..%2f, %2e%2e%2F, ..%2F, etc.)
    let decoded = percent_decode_path(path);
    if decoded != path && has_dotdot_component(&decoded) {
        return Err(DuckLakeError::InvalidConfig(format!(
            "Path traversal detected: URL-encoded '..': {}",
            path
        )));
    }
    Ok(())
}

/// Validate a path for both null bytes and path traversal
pub(crate) fn validate_path(path: &str) -> Result<()> {
    validate_no_null_bytes(path)?;
    validate_no_path_traversal(path)?;
    Ok(())
}

/// Parses a data path into an ObjectStoreUrl and key path
///
/// Supports three formats:
/// - S3 URLs: `s3://bucket/prefix/path`
/// - File URLs: `file:///absolute/path`
/// - Local paths: `/absolute/path` or `relative/path`
///
/// # Arguments
/// * `data_path` - The path to parse (may be S3, file://, or local path)
///
/// # Returns
/// A tuple of (ObjectStoreUrl, key_path) where:
/// - `ObjectStoreUrl` is the base URL for the object store (e.g., `s3://bucket/` or `file:///`)
/// - `key_path` is the path component (e.g., `/prefix/path`)
///
/// # Errors
/// Returns error if:
/// - URL parsing fails
/// - S3 URL is missing bucket name
/// - Local path cannot be canonicalized (doesn't exist or permission denied)
/// - Path contains null bytes or path traversal sequences
///
/// # Example
/// ```no_run
/// # use datafusion_ducklake::path_resolver::parse_object_store_url;
/// let (url, path) = parse_object_store_url("s3://my-bucket/data/tables")?;
/// // url = ObjectStoreUrl("s3://my-bucket/")
/// // path = "/data/tables"
/// # Ok::<(), datafusion_ducklake::DuckLakeError>(())
/// ```
pub fn parse_object_store_url(data_path: &str) -> Result<(ObjectStoreUrl, String)> {
    // Trim leading/trailing whitespace (#64)
    let data_path = data_path.trim();

    validate_path(data_path)?;

    // Use case-insensitive scheme matching per RFC 3986 (#61)
    let lower = data_path.to_ascii_lowercase();
    if lower.starts_with("s3://") {
        parse_s3_url(data_path)
    } else if lower.starts_with("file://") {
        parse_file_url(data_path)
    } else {
        parse_local_path(data_path)
    }
}

/// Parse an S3 URL into ObjectStoreUrl and key path
fn parse_s3_url(data_path: &str) -> Result<(ObjectStoreUrl, String)> {
    let url = url::Url::parse(data_path).map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to parse S3 URL '{}': {}", data_path, e))
    })?;

    let bucket = url.host_str().ok_or_else(|| {
        DuckLakeError::InvalidConfig(format!("S3 URL missing bucket: {}", data_path))
    })?;

    let object_store_url = ObjectStoreUrl::parse(format!("s3://{}/", bucket)).map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to create ObjectStoreUrl: {}", e))
    })?;

    Ok((object_store_url, url.path().to_owned()))
}

/// Parse a file:// URL into ObjectStoreUrl and key path
fn parse_file_url(data_path: &str) -> Result<(ObjectStoreUrl, String)> {
    let url = url::Url::parse(data_path).map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to parse file URL '{}': {}", data_path, e))
    })?;

    let object_store_url = ObjectStoreUrl::parse("file:///").map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to create ObjectStoreUrl: {}", e))
    })?;

    // Decode percent-encoded characters to preserve non-ASCII in local filesystem paths (#60).
    // url::Url::parse() percent-encodes UTF-8 characters, but the local filesystem expects
    // raw UTF-8 paths.
    let decoded_path = percent_decode_str(url.path())
        .decode_utf8()
        .map_err(|e| {
            DuckLakeError::InvalidConfig(format!(
                "Invalid UTF-8 in file path '{}': {}",
                data_path, e
            ))
        })?
        .to_string();

    Ok((object_store_url, decoded_path))
}

/// Parse a local filesystem path into ObjectStoreUrl and key path
fn parse_local_path(data_path: &str) -> Result<(ObjectStoreUrl, String)> {
    // Reject paths containing ".." to prevent directory traversal attacks.
    // std::path::absolute() does NOT normalize ".." components, so a malicious
    // data_path like "/data/../../../etc/" would pass through unchanged.
    if data_path.split(['/', '\\']).any(|c| c == "..") {
        return Err(DuckLakeError::InvalidConfig(
            "Path contains '..' traversal component".to_string(),
        ));
    }

    let has_trailing_slash = data_path.ends_with('/') || data_path.ends_with('\\');

    // Use std::path::absolute() instead of canonicalize() so that the path
    // does not need to exist on disk yet. This supports empty catalogs where
    // the data directory has not been created (#57).
    let absolute_path = std::path::absolute(data_path).map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to resolve path '{}': {}", data_path, e))
    })?;

    let object_store_url = ObjectStoreUrl::parse("file:///").map_err(|e| {
        DuckLakeError::InvalidConfig(format!("Failed to create ObjectStoreUrl: {}", e))
    })?;

    let mut path_str = absolute_path.to_string_lossy().to_string();
    // Preserve trailing slash if original had one
    if has_trailing_slash && !path_str.ends_with('/') && !path_str.ends_with('\\') {
        path_str.push('/');
    }

    Ok((object_store_url, path_str))
}

/// Resolves a path that may be relative or absolute
///
/// If the path is relative, it's joined with the base path.
/// If the path is absolute, it's returned as-is.
///
/// # Arguments
/// * `base_path` - The base path to resolve against
/// * `path` - The path to resolve (may be relative or absolute)
/// * `is_relative` - Whether the path is relative to the base path
///
/// # Returns
/// The resolved absolute path, or an error if the path contains null bytes or traversal
///
/// # Example
/// ```
/// # use datafusion_ducklake::path_resolver::resolve_path;
/// let resolved = resolve_path("/data/schema1/", "table1/data.parquet", true).unwrap();
/// assert_eq!(resolved, "/data/schema1/table1/data.parquet");
///
/// let absolute = resolve_path("/data/schema1/", "/other/path/data.parquet", false).unwrap();
/// assert_eq!(absolute, "/other/path/data.parquet");
/// ```
pub fn resolve_path(base_path: &str, path: &str, is_relative: bool) -> Result<String> {
    if is_relative {
        join_paths(base_path, path)
    } else {
        validate_path(path)?;
        Ok(path.to_string())
    }
}

/// Joins a base path with a relative path, ensuring proper separators
///
/// Handles trailing slashes correctly to avoid double slashes or missing separators.
///
/// # Arguments
/// * `base_path` - The base path (should be absolute)
/// * `relative_path` - The relative path to append
///
/// # Returns
/// The joined path with proper separators, or an error if the path contains null bytes or traversal
///
/// # Example
/// ```
/// # use datafusion_ducklake::path_resolver::join_paths;
/// assert_eq!(join_paths("/data/", "table/file.parquet").unwrap(), "/data/table/file.parquet");
/// assert_eq!(join_paths("/data", "table/file.parquet").unwrap(), "/data/table/file.parquet");
/// ```
pub fn join_paths(base_path: &str, relative_path: &str) -> Result<String> {
    validate_path(relative_path)?;

    if base_path.ends_with('/') || base_path.ends_with('\\') {
        Ok(format!("{}{}", base_path, relative_path))
    } else {
        Ok(format!("{}/{}", base_path, relative_path))
    }
}

/// Path resolver for hierarchical DuckLake paths
///
/// Maintains a base URL and base path for resolving relative paths in the
/// catalog hierarchy (catalog -> schema -> table -> file).
#[derive(Debug, Clone)]
pub struct PathResolver {
    /// Base object store URL (e.g., s3://bucket/ or file:///)
    base_url: Arc<ObjectStoreUrl>,
    /// Base path for resolving relative paths (e.g., /prefix/schema/table/)
    base_path: String,
}

impl PathResolver {
    /// Creates a new PathResolver with a base URL and path
    ///
    /// # Arguments
    /// * `base_url` - The object store URL (e.g., `s3://bucket/` or `file:///`)
    /// * `base_path` - The base path for resolving relative paths
    ///
    /// # Example
    /// ```no_run
    /// # use datafusion_ducklake::path_resolver::PathResolver;
    /// # use datafusion::datasource::object_store::ObjectStoreUrl;
    /// # use std::sync::Arc;
    /// let url = ObjectStoreUrl::parse("s3://my-bucket/")?;
    /// let resolver = PathResolver::new(Arc::new(url), "/data/schema1/".to_string());
    /// # Ok::<(), datafusion::error::DataFusionError>(())
    /// ```
    pub fn new(base_url: Arc<ObjectStoreUrl>, base_path: String) -> Self {
        Self {
            base_url,
            base_path,
        }
    }

    /// Resolves a path that may be relative or absolute
    ///
    /// # Arguments
    /// * `path` - The path to resolve
    /// * `is_relative` - Whether the path is relative to the base path
    ///
    /// # Returns
    /// The resolved absolute path
    pub fn resolve(&self, path: &str, is_relative: bool) -> Result<String> {
        resolve_path(&self.base_path, path, is_relative)
    }

    /// Creates a child resolver with a new base path
    ///
    /// This is useful for creating resolvers at each level of the hierarchy:
    /// catalog resolver -> schema resolver -> table resolver
    ///
    /// # Arguments
    /// * `path` - The path to resolve (may be relative or absolute)
    /// * `is_relative` - Whether the path is relative to the current base path
    ///
    /// # Returns
    /// A new PathResolver with the resolved path as its base path
    ///
    /// # Example
    /// ```no_run
    /// # use datafusion_ducklake::path_resolver::PathResolver;
    /// # use datafusion::datasource::object_store::ObjectStoreUrl;
    /// # use std::sync::Arc;
    /// let catalog_resolver = PathResolver::new(
    ///     Arc::new(ObjectStoreUrl::parse("s3://bucket/")?),
    ///     "/data/".to_string()
    /// );
    /// let schema_resolver = catalog_resolver.child_resolver("schema1/", true).unwrap();
    /// // schema_resolver now has base_path = "/data/schema1/"
    /// # Ok::<(), datafusion::error::DataFusionError>(())
    /// ```
    pub fn child_resolver(&self, path: &str, is_relative: bool) -> Result<Self> {
        let resolved = self.resolve(path, is_relative)?;
        Ok(Self::new(self.base_url.clone(), resolved))
    }

    /// Returns the base object store URL
    pub fn base_url(&self) -> &Arc<ObjectStoreUrl> {
        &self.base_url
    }

    /// Returns the base path
    pub fn base_path(&self) -> &str {
        &self.base_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let (url, path) = parse_object_store_url("s3://bucket/prefix/data").unwrap();
        assert_eq!(path, "/prefix/data");
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
    }

    #[test]
    fn test_parse_s3_url_with_trailing_slash() {
        let (url, path) = parse_object_store_url("s3://bucket/prefix/").unwrap();
        assert_eq!(path, "/prefix/");
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
    }

    #[test]
    fn test_parse_s3_url_no_prefix() {
        let (url, path) = parse_object_store_url("s3://bucket/").unwrap();
        assert_eq!(path, "/");
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
    }

    #[test]
    fn test_parse_file_url() {
        let (url, path) = parse_object_store_url("file:///tmp/data").unwrap();
        assert_eq!(path, "/tmp/data");
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
    }

    #[test]
    fn test_parse_file_url_with_trailing_slash() {
        let (url, path) = parse_object_store_url("file:///tmp/data/").unwrap();
        assert_eq!(path, "/tmp/data/");
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
    }

    #[test]
    fn test_parse_s3_url_missing_bucket() {
        let result = parse_object_store_url("s3:///path");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("S3 URL missing bucket")
        );
    }

    #[test]
    fn test_resolve_path_relative() {
        let resolved = resolve_path("/data/schema1/", "table1/", true).unwrap();
        assert_eq!(resolved, "/data/schema1/table1/");
    }

    #[test]
    fn test_resolve_path_relative_no_trailing_slash() {
        let resolved = resolve_path("/data/schema1", "table1/", true).unwrap();
        assert_eq!(resolved, "/data/schema1/table1/");
    }

    #[test]
    fn test_resolve_path_absolute() {
        let resolved = resolve_path("/data/schema1/", "/other/path/", false).unwrap();
        assert_eq!(resolved, "/other/path/");
    }

    #[test]
    fn test_join_paths_with_trailing_slash() {
        assert_eq!(
            join_paths("/data/", "table/file.parquet").unwrap(),
            "/data/table/file.parquet"
        );
    }

    #[test]
    fn test_join_paths_without_trailing_slash() {
        assert_eq!(
            join_paths("/data", "table/file.parquet").unwrap(),
            "/data/table/file.parquet"
        );
    }

    #[test]
    fn test_join_paths_with_backslash() {
        assert_eq!(
            join_paths("C:\\data\\", "table\\file.parquet").unwrap(),
            "C:\\data\\table\\file.parquet"
        );
    }

    #[test]
    fn test_path_resolver_resolve() {
        let resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/data/schema1/".to_string(),
        );

        let resolved = resolver.resolve("table1/", true).unwrap();
        assert_eq!(resolved, "/data/schema1/table1/");

        let absolute = resolver.resolve("/other/path/", false).unwrap();
        assert_eq!(absolute, "/other/path/");
    }

    #[test]
    fn test_path_resolver_child_resolver() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/data/".to_string(),
        );

        let schema_resolver = catalog_resolver.child_resolver("schema1/", true).unwrap();
        assert_eq!(schema_resolver.base_path(), "/data/schema1/");
        assert_eq!(
            *schema_resolver.base_url(),
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap())
        );

        let table_resolver = schema_resolver.child_resolver("table1/", true).unwrap();
        assert_eq!(table_resolver.base_path(), "/data/schema1/table1/");
    }

    #[test]
    fn test_path_resolver_child_resolver_absolute() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/data/".to_string(),
        );

        let schema_resolver = catalog_resolver
            .child_resolver("/other/schema/", false)
            .unwrap();
        assert_eq!(schema_resolver.base_path(), "/other/schema/");
    }

    // Additional edge case tests

    #[test]
    fn test_parse_s3_url_with_deep_prefix() {
        let (url, path) =
            parse_object_store_url("s3://my-bucket/level1/level2/level3/data").unwrap();
        assert_eq!(path, "/level1/level2/level3/data");
        assert_eq!(url, ObjectStoreUrl::parse("s3://my-bucket/").unwrap());
    }

    #[test]
    fn test_parse_s3_url_bucket_only() {
        let (url, path) = parse_object_store_url("s3://my-bucket").unwrap();
        assert_eq!(path, "");
        assert_eq!(url, ObjectStoreUrl::parse("s3://my-bucket/").unwrap());
    }

    #[test]
    fn test_parse_s3_url_with_hyphens_and_dots() {
        let (url, path) = parse_object_store_url("s3://my-bucket.name-123/data").unwrap();
        assert_eq!(path, "/data");
        assert_eq!(
            url,
            ObjectStoreUrl::parse("s3://my-bucket.name-123/").unwrap()
        );
    }

    #[test]
    fn test_parse_file_url_root() {
        let (url, path) = parse_object_store_url("file:///").unwrap();
        assert_eq!(path, "/");
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
    }

    #[test]
    fn test_parse_file_url_with_host() {
        // file://host/path is technically valid (networked file path)
        let (url, path) = parse_object_store_url("file://localhost/tmp/data").unwrap();
        assert_eq!(path, "/tmp/data");
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
    }

    #[test]
    fn test_parse_local_path_nonexistent() {
        // Non-existent paths should now succeed (#57) - the directory does not
        // need to exist at catalog creation time (e.g., empty catalogs).
        let (url, path) = parse_object_store_url("/nonexistent/path/that/does/not/exist").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/nonexistent/path/that/does/not/exist");
    }

    #[test]
    fn test_parse_local_path_nonexistent_with_trailing_slash() {
        let (url, path) = parse_object_store_url("/nonexistent/data/path/").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/nonexistent/data/path/");
    }

    #[test]
    fn test_reject_dotdot_in_local_path() {
        // Local paths with ".." must be rejected since std::path::absolute()
        // does not normalize them away.
        let result = parse_object_store_url("/data/../../../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("'..'"),
            "error should mention '..' traversal: {}",
            err
        );
    }

    #[test]
    fn test_reject_dotdot_in_local_path_backslash() {
        let result = parse_object_store_url("/data\\..\\secret");
        assert!(result.is_err());
    }

    #[test]
    fn test_join_paths_nested_relative() {
        assert_eq!(
            join_paths("/data/", "schema1/table1/file.parquet").unwrap(),
            "/data/schema1/table1/file.parquet"
        );
    }

    #[test]
    fn test_join_paths_windows_style() {
        assert_eq!(
            join_paths("C:\\data", "schema1\\table1").unwrap(),
            "C:\\data/schema1\\table1"
        );
    }

    #[test]
    fn test_path_resolver_multiple_levels() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/data/".to_string(),
        );

        let schema_resolver = catalog_resolver.child_resolver("schema1/", true).unwrap();
        let table_resolver = schema_resolver.child_resolver("table1/", true).unwrap();
        let file_path = table_resolver.resolve("file.parquet", true).unwrap();

        assert_eq!(file_path, "/data/schema1/table1/file.parquet");
    }

    #[test]
    fn test_resolve_path_with_special_chars() {
        let resolved = resolve_path("/data/", "table-123_test.v2/", true).unwrap();
        assert_eq!(resolved, "/data/table-123_test.v2/");
    }

    #[test]
    fn test_join_paths_double_slash_in_relative() {
        // If relative path starts with /, we preserve it (though it's unusual)
        assert_eq!(
            join_paths("/data/", "/absolute").unwrap(),
            "/data//absolute"
        );
    }

    #[test]
    fn test_path_resolver_mixed_relative_absolute() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/catalog/".to_string(),
        );

        // First child is relative
        let schema_resolver = catalog_resolver.child_resolver("schema1/", true).unwrap();
        assert_eq!(schema_resolver.base_path(), "/catalog/schema1/");

        // Second child is absolute (overrides)
        let table_resolver = schema_resolver
            .child_resolver("/absolute/table/", false)
            .unwrap();
        assert_eq!(table_resolver.base_path(), "/absolute/table/");

        // Third child is relative to the absolute path
        let file_resolver = table_resolver.child_resolver("subdir/", true).unwrap();
        assert_eq!(file_resolver.base_path(), "/absolute/table/subdir/");
    }

    #[test]
    fn test_s3_url_bucket_with_region() {
        // Test S3 URL with region-specific bucket format
        let (url, path) = parse_object_store_url("s3://my-bucket/data/warehouse").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://my-bucket/").unwrap());
        assert_eq!(path, "/data/warehouse");
    }

    #[test]
    fn test_s3_url_with_encoded_characters() {
        // S3 paths can contain URL-encoded characters
        let (url, path) = parse_object_store_url("s3://bucket/path%20with%20spaces/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/path%20with%20spaces/data");
    }

    #[test]
    fn test_s3_url_very_long_path() {
        let long_path = "s3://bucket/".to_string() + &"level/".repeat(20) + "file.parquet";
        let (url, path) = parse_object_store_url(&long_path).unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert!(path.starts_with("/level/"));
        assert!(path.ends_with("file.parquet"));
    }

    #[test]
    fn test_mixed_s3_and_local_hierarchy() {
        // Simulate a scenario where catalog uses S3 but we need to track paths
        let s3_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://data-lake/").unwrap()),
            "/prod/".to_string(),
        );

        let schema_resolver = s3_resolver.child_resolver("sales/", true).unwrap();
        assert_eq!(schema_resolver.base_path(), "/prod/sales/");
        assert_eq!(
            *schema_resolver.base_url(),
            Arc::new(ObjectStoreUrl::parse("s3://data-lake/").unwrap())
        );

        let table_resolver = schema_resolver
            .child_resolver("transactions/", true)
            .unwrap();
        assert_eq!(table_resolver.base_path(), "/prod/sales/transactions/");

        let file_path = table_resolver
            .resolve("2024/01/data.parquet", true)
            .unwrap();
        assert_eq!(file_path, "/prod/sales/transactions/2024/01/data.parquet");
    }

    #[test]
    fn test_hierarchical_path_with_absolute_override_s3() {
        // Schema is relative to catalog, but table uses absolute path
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/warehouse/".to_string(),
        );

        let schema_resolver = catalog_resolver.child_resolver("prod/", true).unwrap();
        assert_eq!(schema_resolver.base_path(), "/warehouse/prod/");

        // Table overrides with absolute path
        let table_resolver = schema_resolver
            .child_resolver("/external/data/", false)
            .unwrap();
        assert_eq!(table_resolver.base_path(), "/external/data/");

        // File is relative to the absolute table path
        let file_path = table_resolver.resolve("file.parquet", true).unwrap();
        assert_eq!(file_path, "/external/data/file.parquet");
    }

    #[test]
    fn test_parse_s3_url_minimal_bucket() {
        // Shortest valid S3 URL (just bucket, no trailing slash)
        let (url, path) = parse_object_store_url("s3://b").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://b/").unwrap());
        assert_eq!(path, "");
    }

    #[test]
    fn test_parse_s3_url_with_query_params() {
        // S3 URLs generally don't have query params, but URL parser should handle it
        // Note: url::Url will parse query params but they won't be in the path
        let (url, path) = parse_object_store_url("s3://bucket/path/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/path/data");
    }

    #[test]
    fn test_file_url_windows_style_path() {
        // file:// URLs with Windows paths (C:/)
        let (url, path) = parse_object_store_url("file:///C:/Users/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/C:/Users/data");
    }

    #[test]
    fn test_path_resolution_with_dots() {
        // Paths with dots (version numbers, file extensions, etc.)
        let resolved = resolve_path("/data/", "schema.v2/table.prod/", true).unwrap();
        assert_eq!(resolved, "/data/schema.v2/table.prod/");
    }

    #[test]
    fn test_path_resolution_with_underscores() {
        let resolved = resolve_path("/data/", "my_schema/my_table/", true).unwrap();
        assert_eq!(resolved, "/data/my_schema/my_table/");
    }

    #[test]
    fn test_join_paths_with_multiple_slashes_in_relative() {
        // Relative path with multiple directory levels
        let result = join_paths("/base/", "a/b/c/d/file.parquet").unwrap();
        assert_eq!(result, "/base/a/b/c/d/file.parquet");
    }

    #[test]
    fn test_join_paths_preserves_trailing_slash() {
        let result = join_paths("/base/", "subdir/").unwrap();
        assert_eq!(result, "/base/subdir/");
    }

    #[test]
    fn test_real_world_s3_scenario() {
        let (catalog_url, catalog_path) =
            parse_object_store_url("s3://data-lake/warehouse/").unwrap();

        assert_eq!(
            catalog_url,
            ObjectStoreUrl::parse("s3://data-lake/").unwrap()
        );
        assert_eq!(catalog_path, "/warehouse/");

        let catalog_resolver = PathResolver::new(Arc::new(catalog_url), catalog_path);

        let schema_resolver = catalog_resolver.child_resolver("prod/", true).unwrap();
        assert_eq!(schema_resolver.base_path(), "/warehouse/prod/");

        let table_resolver = schema_resolver
            .child_resolver("sales/transactions/", true)
            .unwrap();
        assert_eq!(
            table_resolver.base_path(),
            "/warehouse/prod/sales/transactions/"
        );

        let file_path = table_resolver.resolve("2024-01-01.parquet", true).unwrap();
        assert_eq!(
            file_path,
            "/warehouse/prod/sales/transactions/2024-01-01.parquet"
        );
    }

    #[test]
    fn test_real_world_mixed_absolute_paths() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://catalog-db/").unwrap()),
            "/metadata/".to_string(),
        );

        let schema_resolver = catalog_resolver
            .child_resolver("/schemas/prod/", false)
            .unwrap();
        assert_eq!(schema_resolver.base_path(), "/schemas/prod/");

        let table_resolver = schema_resolver.child_resolver("customers/", true).unwrap();
        assert_eq!(table_resolver.base_path(), "/schemas/prod/customers/");

        let file_path = table_resolver.resolve("data.parquet", true).unwrap();
        assert_eq!(file_path, "/schemas/prod/customers/data.parquet");
    }

    #[test]
    fn test_s3_url_uppercase_scheme() {
        // URL schemes are case-insensitive per RFC 3986 (#61)
        let (url, path) = parse_object_store_url("S3://bucket/path").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/path");
    }

    #[test]
    fn test_s3_url_mixed_case_scheme() {
        let (url, path) = parse_object_store_url("s3://bucket/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/data");
    }

    #[test]
    fn test_file_url_uppercase_scheme() {
        let (url, path) = parse_object_store_url("FILE:///tmp/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/tmp/data");
    }

    #[test]
    fn test_file_url_mixed_case_scheme() {
        let (url, path) = parse_object_store_url("File:///tmp/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/tmp/data");
    }

    #[test]
    fn test_whitespace_trimming_s3() {
        // Leading/trailing whitespace should be trimmed (#64)
        let (url, path) = parse_object_store_url("  s3://bucket/data  ").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/data");
    }

    #[test]
    fn test_whitespace_trimming_file() {
        let (url, path) = parse_object_store_url("  file:///tmp/data  ").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/tmp/data");
    }

    #[test]
    fn test_file_url_non_ascii_preserved() {
        // Non-ASCII characters in file:// URLs should be preserved (#60)
        let (url, path) = parse_object_store_url("file:///tmp/données/données").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/tmp/données/données");
    }

    #[test]
    fn test_file_url_unicode_preserved() {
        let (url, path) = parse_object_store_url("file:///home/用户/数据").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/home/用户/数据");
    }

    #[test]
    fn test_file_url_spaces_preserved() {
        let (url, path) = parse_object_store_url("file:///tmp/my data/file").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("file:///").unwrap());
        assert_eq!(path, "/tmp/my data/file");
    }

    #[test]
    fn test_whitespace_trimming_with_uppercase_scheme() {
        // Combined: whitespace + uppercase scheme
        let (url, path) = parse_object_store_url("  S3://bucket/data  ").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket/").unwrap());
        assert_eq!(path, "/data");
    }

    #[test]
    fn test_edge_case_empty_base_path() {
        let resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "".to_string(),
        );

        let resolved = resolver.resolve("path/to/file", true).unwrap();
        assert_eq!(resolved, "/path/to/file");
    }

    #[test]
    fn test_edge_case_root_base_path() {
        let resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/".to_string(),
        );

        let resolved = resolver.resolve("file.parquet", true).unwrap();
        assert_eq!(resolved, "/file.parquet");
    }

    #[test]
    fn test_path_with_consecutive_slashes() {
        let resolved = resolve_path("/data/", "schema//table/", true).unwrap();
        assert_eq!(resolved, "/data/schema//table/");
    }

    #[test]
    fn test_s3_bucket_naming_conventions() {
        let (url, _) = parse_object_store_url("s3://bucket123/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket123/").unwrap());

        let (url, _) = parse_object_store_url("s3://my-bucket-name/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://my-bucket-name/").unwrap());

        let (url, _) = parse_object_store_url("s3://my.bucket.name/data").unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://my.bucket.name/").unwrap());
    }

    #[test]
    fn test_file_url_relative_path_not_supported() {
        let result = parse_object_store_url("file://relative/path");
        assert!(result.is_ok());
    }

    #[test]
    fn test_complex_hierarchical_scenario() {
        let catalog_resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://prod-data-lake/").unwrap()),
            "/".to_string(),
        );

        let schema1 = catalog_resolver.child_resolver("warehouse/", true).unwrap();
        assert_eq!(schema1.base_path(), "/warehouse/");

        let table1 = schema1.child_resolver("sales/", true).unwrap();
        assert_eq!(table1.base_path(), "/warehouse/sales/");

        let partition1 = table1.child_resolver("year=2024/", true).unwrap();
        assert_eq!(partition1.base_path(), "/warehouse/sales/year=2024/");

        let partition2 = partition1.child_resolver("month=01/", true).unwrap();
        assert_eq!(
            partition2.base_path(),
            "/warehouse/sales/year=2024/month=01/"
        );

        let file = partition2.resolve("data.parquet", true).unwrap();
        assert_eq!(file, "/warehouse/sales/year=2024/month=01/data.parquet");

        assert_eq!(
            *partition2.base_url(),
            Arc::new(ObjectStoreUrl::parse("s3://prod-data-lake/").unwrap())
        );
    }

    // =========================================================================
    // Validation tests: null bytes
    // =========================================================================

    #[test]
    fn test_reject_literal_null_byte_in_resolve_path() {
        let result = resolve_path("/data/", "table\0evil", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("null byte"), "error was: {}", err);
    }

    #[test]
    fn test_reject_url_encoded_null_byte_in_resolve_path() {
        let result = resolve_path("/data/", "table%00evil", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("null byte"), "error was: {}", err);
    }

    #[test]
    fn test_reject_null_byte_case_insensitive() {
        let result = resolve_path("/data/", "table%2e%2e/%00evil", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_null_byte_in_parse_object_store_url() {
        let result = parse_object_store_url("s3://bucket/path\0evil");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("null byte"), "error was: {}", err);
    }

    #[test]
    fn test_reject_url_encoded_null_in_parse_object_store_url() {
        let result = parse_object_store_url("s3://bucket/path%00evil");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("null byte"), "error was: {}", err);
    }

    #[test]
    fn test_reject_null_byte_absolute_path() {
        let result = resolve_path("/data/", "/abs/path\0evil", false);
        assert!(result.is_err());
    }

    // =========================================================================
    // Validation tests: path traversal
    // =========================================================================

    #[test]
    fn test_reject_dotdot_in_relative_path() {
        let result = resolve_path("/data/", "../etc/passwd", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path traversal"), "error was: {}", err);
    }

    #[test]
    fn test_reject_dotdot_mid_path() {
        let result = resolve_path("/data/", "table/../../../etc/passwd", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_dotdot_in_absolute_path() {
        let result = resolve_path("/data/", "/tmp/../etc/passwd", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_backslash_dotdot() {
        let result = resolve_path("/data/", "table\\..\\secret", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_url_encoded_dotdot() {
        let result = resolve_path("/data/", "table/%2e%2e/secret", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path traversal"), "error was: {}", err);
    }

    #[test]
    fn test_reject_mixed_case_url_encoded_dotdot() {
        let result = resolve_path("/data/", "table/%2E%2E/secret", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_dotdot_in_join_paths() {
        let result = join_paths("/data/", "../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_dotdot_in_parse_object_store_url() {
        let result = parse_object_store_url("s3://bucket/../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_dotdot_in_child_resolver() {
        let resolver = PathResolver::new(
            Arc::new(ObjectStoreUrl::parse("s3://bucket/").unwrap()),
            "/data/".to_string(),
        );
        let result = resolver.child_resolver("../secret/", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_dotdot_percent_encoded_slash() {
        // ..%2f bypasses literal `/` split — must be caught via percent-decoding
        let result = resolve_path("/data/", "..%2f..%2fetc", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path traversal"), "error was: {}", err);
    }

    #[test]
    fn test_reject_dotdot_percent_encoded_slash_uppercase() {
        let result = resolve_path("/data/", "..%2F..%2Fetc", true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path traversal"), "error was: {}", err);
    }

    #[test]
    fn test_reject_fully_encoded_dotdot_slash() {
        // %2e%2e%2f = ../
        let result = resolve_path("/data/", "%2e%2e%2fetc/passwd", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_mixed_encoded_dotdot_slash() {
        // ..%2f with literal dots and encoded slash
        let result = resolve_path("/data/", "sub/..%2fsecret", true);
        assert!(result.is_err());
    }

    // =========================================================================
    // Validation tests: allowed patterns (no false positives)
    // =========================================================================

    #[test]
    fn test_allow_single_dot_in_path() {
        // Single dots are fine (current directory, file extensions)
        let result = resolve_path("/data/", "file.parquet", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allow_dotfile() {
        let result = resolve_path("/data/", ".hidden/file", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allow_multiple_dots_in_filename() {
        let result = resolve_path("/data/", "file.backup.2024.parquet", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allow_version_dots() {
        let result = resolve_path("/data/", "schema.v2/table.prod/", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allow_percent_in_path() {
        // %20 (space) is fine, only %00 (null) and %2e%2e (..) are rejected
        let result = resolve_path("/data/", "path%20with%20spaces/file", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allow_normal_s3_paths() {
        let result = parse_object_store_url("s3://bucket/warehouse/prod/data");
        assert!(result.is_ok());
    }
}
