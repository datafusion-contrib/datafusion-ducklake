//! Parquet footer metadata helpers shared between read and write paths.

/// Convert a DuckLake-spec `footer_size` (thrift metadata length only) into the
/// value to pass to parquet's `with_metadata_size_hint`.
///
/// The parquet reader interprets the hint as "bytes to prefetch from end of
/// file" and needs `thrift_metadata_len + 8` (length field + `PAR1` magic) to
/// land the entire footer in a single read. Without the +8, the reader has to
/// issue a second I/O — correctness preserved, but the optimization that
/// motivated storing the hint is defeated.
///
/// The catalog stores the bare thrift length (per DuckLake spec) so that
/// cross-engine reads via the DuckDB DuckLake extension pass its footer-size
/// validation. This helper bridges the two semantics.
///
/// Returns `None` if the input is missing, non-positive, or its `+ 8` overflows
/// `usize`. Callers should treat `None` as "no hint" (parquet falls back to a
/// default probe), not as an error.
pub(crate) fn metadata_size_hint_from_footer(footer_size: Option<i64>) -> Option<usize> {
    let size = footer_size?;
    if size <= 0 {
        return None;
    }
    usize::try_from(size).ok()?.checked_add(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_trailer() {
        // The stored value is the DuckLake-spec thrift metadata length only.
        // The parquet prefetch hint needs the +8 trailer to land the full
        // footer in one read.
        assert_eq!(metadata_size_hint_from_footer(Some(716)), Some(724));
        assert_eq!(metadata_size_hint_from_footer(Some(1)), Some(9));
    }

    #[test]
    fn rejects_invalid() {
        assert_eq!(metadata_size_hint_from_footer(None), None);
        assert_eq!(metadata_size_hint_from_footer(Some(0)), None);
        assert_eq!(metadata_size_hint_from_footer(Some(-1)), None);
        // On 32-bit targets where `usize::MAX < i64::MAX`, the conversion fails
        // safely (returns None) rather than wrapping; the `checked_add` defends
        // against the analogous overflow on any future wider-pointer platform.
        // On 64-bit `i64::MAX` fits cleanly so we don't assert that boundary.
    }
}
