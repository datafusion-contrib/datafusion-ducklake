//! Per-file sort attestation: the writer checks that the rows it writes
//! really are ordered by the table's sort key, the result is recorded per data
//! file (the experimental `ext_ducklake_file_order` sidecar, see
//! `crate::ext_file_order`), and a scan declares that ordering to DataFusion
//! only for files with a record.
//!
//! Why not the catalog's sort spec: in DuckLake `SET SORTED BY` is a
//! write-time hint. No catalog row records which spec (if any) a data file was
//! written under; `sort_on_insert = false` writes unsorted files under a live
//! spec; `ducklake_add_data_files` registers arbitrary files; inlined rows are
//! unsorted until flushed; and a spec change leaves older files as they were.
//! Declaring an ordering that is not true returns wrong rows silently, so the
//! claim has to be proven per file.
//!
//! Why not the standard parquet `sorting_columns`: it is fixed when the
//! writer opens, before any row is seen, so setting it means trusting the
//! caller's input order (the streaming write session takes rows in caller
//! order), and other readers trust it blindly. The check here instead observes
//! every row written, so it cannot be wrong whoever produced the input.
//!
//! Keys are recorded by field id, so a renamed column keeps its record and a
//! dropped-and-re-added column (new field id) does not inherit one.

use arrow::compute::SortOptions;

/// One attested sort key: a parquet field id and its direction / null order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttestedKey {
    pub(crate) field_id: i32,
    pub(crate) options: SortOptions,
}

/// `field_id:asc|desc:nulls_first|nulls_last`, comma-separated, primary key first.
#[cfg(feature = "write")]
pub(crate) fn encode(keys: &[AttestedKey]) -> String {
    keys.iter()
        .map(|key| {
            format!(
                "{}:{}:{}",
                key.field_id,
                if key.options.descending {
                    "desc"
                } else {
                    "asc"
                },
                if key.options.nulls_first {
                    "nulls_first"
                } else {
                    "nulls_last"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse [`encode`]'s output. Anything malformed yields `None` (no attestation),
/// never a partial key: a prefix of a garbled value could claim an ordering the
/// writer never checked.
pub(crate) fn decode(value: &str) -> Option<Vec<AttestedKey>> {
    let keys = value
        .split(',')
        .map(|part| {
            let mut it = part.split(':');
            let field_id = it.next()?.parse().ok()?;
            let descending = match it.next()? {
                "asc" => false,
                "desc" => true,
                _ => return None,
            };
            let nulls_first = match it.next()? {
                "nulls_first" => true,
                "nulls_last" => false,
                _ => return None,
            };
            if it.next().is_some() {
                return None;
            }
            Some(AttestedKey {
                field_id,
                options: SortOptions {
                    descending,
                    nulls_first,
                },
            })
        })
        .collect::<Option<Vec<_>>>()?;
    (!keys.is_empty()).then_some(keys)
}

#[cfg(feature = "write")]
pub(crate) use write::{FileOrderRequest, SortCheck};

#[cfg(feature = "write")]
mod write {
    use std::sync::Arc;

    use arrow::array::ArrayRef;
    use arrow::datatypes::Schema;
    use arrow::record_batch::RecordBatch;
    use arrow::row::{OwnedRow, RowConverter, SortField};
    use parquet::file::metadata::SortingColumn;

    use super::{AttestedKey, encode};
    use crate::Result;
    use crate::ext_file_order::{BY_WRITE, FileOrder};
    use crate::metadata_writer::MetadataWriter;
    use crate::sort::{NullOrder, SortDirection, SortSpec};

    /// What a write asks of each file it writes under the `ext_file_order`
    /// option: check the rows against the table's sort key and, when they
    /// follow it, record that in the sidecar as soon as the file is finished.
    /// That can precede the upload: a row for a path whose upload then fails
    /// matches no catalog file and is pruned like any other.
    ///
    /// One pointer wide: writers and sessions holding it live in long-held
    /// futures.
    #[derive(Debug, Clone)]
    pub(crate) struct FileOrderRequest(Arc<RequestInner>);

    #[derive(Debug)]
    struct RequestInner {
        metadata: Arc<dyn MetadataWriter>,
        table_id: i64,
        /// Top-level field indices of the written schema, with the key's
        /// direction and null order.
        columns: Vec<SortingColumn>,
    }

    impl FileOrderRequest {
        /// The request for `spec`'s keys over `schema`'s top-level fields;
        /// `None` when the table is unsorted, a key is an expression this
        /// crate cannot evaluate, or a key is not a field of `schema`.
        pub(crate) fn new(
            metadata: Arc<dyn MetadataWriter>,
            table_id: i64,
            spec: Option<&SortSpec>,
            schema: &Schema,
        ) -> Option<Self> {
            let columns = spec?
                .producible_columns()?
                .into_iter()
                .map(|(name, direction, nulls)| {
                    let (index, _) = schema.column_with_name(&name)?;
                    Some(SortingColumn {
                        column_idx: i32::try_from(index).ok()?,
                        descending: direction == SortDirection::Desc,
                        nulls_first: nulls == NullOrder::NullsFirst,
                    })
                })
                .collect::<Option<Vec<_>>>()
                .filter(|columns| !columns.is_empty())?;
            Some(Self(Arc::new(RequestInner {
                metadata,
                table_id,
                columns,
            })))
        }

        /// A fresh check for one file written with `schema`, which carries
        /// the request's columns at the same indices plus their field ids.
        pub(crate) fn check(&self, schema: &Schema) -> Option<SortCheck> {
            SortCheck::new(schema, &self.0.columns)
        }

        /// Record that the finished file at catalog `path` follows `sort_key`.
        /// Best effort: the data write never fails over its sidecar row, and a
        /// file without one reads as unordered, as it always did.
        pub(crate) fn record(&self, path: &str, file_size_bytes: i64, sort_key: String) {
            let order = FileOrder {
                path: path.to_string(),
                file_size_bytes,
                sort_key,
            };
            if let Err(error) =
                self.0
                    .metadata
                    .record_file_orders(self.0.table_id, BY_WRITE, &[order])
            {
                tracing::warn!(
                    %error,
                    path,
                    "ext_file_order: not recorded; the file reads as unordered"
                );
            }
        }
    }

    /// Watches the rows written to one file and reports, at close, whether they
    /// were in order on the requested key.
    #[derive(Debug)]
    pub(crate) struct SortCheck {
        columns: Vec<usize>,
        keys: Vec<AttestedKey>,
        converter: RowConverter,
        last: Option<OwnedRow>,
        sorted: bool,
        rows: usize,
    }

    impl SortCheck {
        /// A check for `requested`, whose `column_idx` is a top-level field index
        /// of `schema` (the file's schema, carrying `PARQUET:field_id`s). `None`
        /// when any key has no field id or a type the row format cannot compare,
        /// in which case the file is simply written without an attestation.
        pub(crate) fn new(schema: &Schema, requested: &[SortingColumn]) -> Option<Self> {
            let field_ids: Option<Vec<i32>> = requested
                .iter()
                .map(|column| {
                    let field = schema
                        .fields()
                        .get(usize::try_from(column.column_idx).ok()?)?;
                    field.metadata().get("PARQUET:field_id")?.parse().ok()
                })
                .collect();
            Self::with_field_ids(schema, requested, &field_ids?)
        }

        /// [`Self::new`] for a file whose columns carry no field ids (matched by
        /// name, as the reader matches them): `field_ids` are the catalog ids
        /// the requested columns resolve to, one per requested column.
        pub(crate) fn with_field_ids(
            schema: &Schema,
            requested: &[SortingColumn],
            field_ids: &[i32],
        ) -> Option<Self> {
            if requested.is_empty() || requested.len() != field_ids.len() {
                return None;
            }
            let mut columns = Vec::with_capacity(requested.len());
            let mut keys = Vec::with_capacity(requested.len());
            let mut fields = Vec::with_capacity(requested.len());
            for (column, field_id) in requested.iter().zip(field_ids) {
                let index = usize::try_from(column.column_idx).ok()?;
                let field = schema.fields().get(index)?;
                let field_id = *field_id;
                let options = arrow::compute::SortOptions {
                    descending: column.descending,
                    nulls_first: column.nulls_first,
                };
                columns.push(index);
                keys.push(AttestedKey {
                    field_id,
                    options,
                });
                fields.push(SortField::new_with_options(
                    field.data_type().clone(),
                    options,
                ));
            }
            let converter = RowConverter::new(fields).ok()?;
            Some(Self {
                columns,
                keys,
                converter,
                last: None,
                sorted: true,
                rows: 0,
            })
        }

        /// Feed the next batch, in write order.
        pub(crate) fn observe(&mut self, batch: &RecordBatch) -> Result<()> {
            if !self.sorted || batch.num_rows() == 0 {
                return Ok(());
            }
            let arrays: Vec<ArrayRef> = self
                .columns
                .iter()
                .map(|&index| batch.column(index).clone())
                .collect();
            let rows = self.converter.convert_columns(&arrays)?;
            // Row-format bytes compare exactly as the requested `SortOptions`
            // order, nulls included, so "in order" is one memcmp per adjacent pair.
            if let Some(last) = &self.last
                && rows.row(0) < last.row()
            {
                self.sorted = false;
                return Ok(());
            }
            for i in 1..rows.num_rows() {
                if rows.row(i) < rows.row(i - 1) {
                    self.sorted = false;
                    return Ok(());
                }
            }
            self.last = Some(rows.row(rows.num_rows() - 1).owned());
            self.rows += batch.num_rows();
            Ok(())
        }

        /// The footer value to record, when every row written was in order.
        pub(crate) fn attestation(&self) -> Option<String> {
            (self.sorted && self.rows > 0).then(|| encode(&self.keys))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "write")]
    #[test]
    fn encode_decode_round_trip() {
        let keys = vec![
            AttestedKey {
                field_id: 1,
                options: SortOptions {
                    descending: true,
                    nulls_first: true,
                },
            },
            AttestedKey {
                field_id: 7,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ];
        let encoded = encode(&keys);
        assert_eq!(encoded, "1:desc:nulls_first,7:asc:nulls_last");
        assert_eq!(decode(&encoded), Some(keys));
    }

    #[test]
    fn malformed_values_attest_nothing() {
        for value in [
            "",
            "x:asc:nulls_last",
            "1:up:nulls_last",
            "1:asc",
            "1:asc:nulls_last:9",
            "1:asc:nulls_last,",
        ] {
            assert_eq!(decode(value), None, "{value:?}");
        }
    }

    #[cfg(feature = "write")]
    mod check {
        use std::collections::HashMap;
        use std::sync::Arc;

        use arrow::array::{Int64Array, TimestampMicrosecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use arrow::record_batch::RecordBatch;
        use parquet::file::metadata::SortingColumn;

        use super::super::SortCheck;

        fn schema() -> Arc<Schema> {
            let id = |n: &str| HashMap::from([("PARQUET:field_id".to_string(), n.to_string())]);
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Int64, true).with_metadata(id("1")),
                Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true)
                    .with_metadata(id("2")),
            ]))
        }

        fn batch(ts: Vec<Option<i64>>) -> RecordBatch {
            let n = ts.len() as i64;
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values(0..n)),
                    Arc::new(TimestampMicrosecondArray::from(ts)),
                ],
            )
            .unwrap()
        }

        fn desc_ts() -> Vec<SortingColumn> {
            vec![SortingColumn {
                column_idx: 1,
                descending: true,
                nulls_first: true,
            }]
        }

        #[test]
        fn sorted_across_batches_attests() {
            let mut check = SortCheck::new(&schema(), &desc_ts()).unwrap();
            check
                .observe(&batch(vec![None, Some(9), Some(9), Some(5)]))
                .unwrap();
            check.observe(&batch(vec![Some(5), Some(1)])).unwrap();
            assert_eq!(check.attestation().as_deref(), Some("2:desc:nulls_first"));
        }

        #[test]
        fn disorder_inside_a_batch_attests_nothing() {
            let mut check = SortCheck::new(&schema(), &desc_ts()).unwrap();
            check.observe(&batch(vec![Some(9), Some(10)])).unwrap();
            assert_eq!(check.attestation(), None);
        }

        #[test]
        fn disorder_across_a_batch_boundary_attests_nothing() {
            let mut check = SortCheck::new(&schema(), &desc_ts()).unwrap();
            check.observe(&batch(vec![Some(9), Some(5)])).unwrap();
            check.observe(&batch(vec![Some(6), Some(1)])).unwrap();
            assert_eq!(check.attestation(), None);
        }

        #[test]
        fn null_on_the_wrong_side_attests_nothing() {
            // NULLS FIRST requested: a null after a value breaks the order.
            let mut check = SortCheck::new(&schema(), &desc_ts()).unwrap();
            check.observe(&batch(vec![Some(9), None])).unwrap();
            assert_eq!(check.attestation(), None);
        }

        #[test]
        fn empty_file_attests_nothing() {
            let check = SortCheck::new(&schema(), &desc_ts()).unwrap();
            assert_eq!(check.attestation(), None);
        }

        #[test]
        fn column_without_field_id_is_not_checked() {
            let plain = Schema::new(vec![Field::new("ts", DataType::Int64, true)]);
            let requested = vec![SortingColumn {
                column_idx: 0,
                descending: false,
                nulls_first: false,
            }];
            assert!(SortCheck::new(&plain, &requested).is_none());
        }
    }
}
