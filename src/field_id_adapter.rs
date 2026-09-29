//! Resolve each data file's columns by parquet field id when the reader opens
//! the file, rather than while the scan is planned.
//!
//! A DuckLake column keeps its `column_id` across renames, and a data file tags
//! each column it stores with that id as its parquet field id. Which physical
//! column backs a catalog column is therefore a property of the file, and
//! official DuckLake binds it per file as each reader opens
//! (`DuckLakeMultiFileReader::Bind` declares `MultiFileColumnMappingMode::BY_FIELD_ID`
//! and `MultiFileReader::CreateMapping` resolves the ids against the opened file's
//! columns). [`FieldIdExprAdapterFactory`] does the same inside DataFusion's parquet
//! opener: the opener hands it the file's Arrow schema, which carries the field ids
//! the reader read from the footer it has just fetched, and the adapter rewrites the
//! scan's projection and predicate from catalog columns to that file's columns.
//! Planning therefore reads no footer, and a scan that stops early never opens the
//! files it does not reach.
//!
//! The per-file resolution is the one [`build_read_schema_with_field_id_mapping_from_schema`]
//! describes: a renamed column is read under its physical name, a column the file
//! does not carry is filled with its `initial_default` or NULL, and a nested column
//! is read with its children resolved by nested field id. [`CatalogTypeExpr`] then
//! presents each value under the catalog's type with the per-column conversion
//! [`crate::column_rename::ColumnRenameExec`] applies.

use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, LargeListArray, ListArray, MapArray, StructArray,
    make_array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, FieldRef, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result as DataFusionResult, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::functions::core::getfield::GetFieldFunc;
use datafusion::logical_expr::ColumnarValue;
use datafusion::logical_expr::ScalarUDF;
use datafusion::physical_expr::expressions::{CastExpr, Column, IsNotNullExpr, Literal};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::column_rename::{coerce_column, types_equal_ignoring_field_metadata};
use crate::metadata_provider::DuckLakeTableColumn;
use crate::row_id::{ROW_ID_PARQUET_FIELD_ID, SNAPSHOT_ID_PARQUET_FIELD_ID};
use crate::types::{
    ABSENT_FIELD_PREFIX, DuckLakeDefaultExprAdapterFactory, arrow_schema_field_ids,
    build_read_schema_with_field_id_mapping_from_schema, parse_ducklake_default_scalar,
};

/// Builds a [`FieldIdExprAdapter`] for each file a scan opens.
///
/// The scan's logical file schema must be the table's physical schema, one field
/// per entry of `columns` in the same order.
#[derive(Debug)]
pub(crate) struct FieldIdExprAdapterFactory {
    columns: Arc<[DuckLakeTableColumn]>,
    row_lineage: Option<Arc<RowLineageProbes>>,
}

/// Names of the scan columns that report a file's embedded lineage columns;
/// each `None` when the scan does not ask for it.
#[derive(Debug, Default)]
pub(crate) struct RowLineageProbes {
    /// Resolves to the file's `_ducklake_internal_row_id` column, found by
    /// [`ROW_ID_PARQUET_FIELD_ID`], or NULL when it stores none.
    pub(crate) embedded: Option<String>,
    /// Resolves to whether the file stores that column.
    pub(crate) has_embedded: Option<String>,
    /// Resolves to the file's `_ducklake_internal_snapshot_id` column, found by
    /// [`SNAPSHOT_ID_PARQUET_FIELD_ID`], or NULL when it stores none: each row's
    /// origin snapshot in a file compaction merged.
    pub(crate) embedded_snapshot: Option<String>,
}

impl FieldIdExprAdapterFactory {
    pub(crate) fn new(columns: &[DuckLakeTableColumn]) -> Self {
        Self {
            columns: columns.into(),
            row_lineage: None,
        }
    }

    /// Also resolve the scan columns named by `probes` per file, as official's
    /// `GetVirtualColumnExpression` resolves a file's row id.
    pub(crate) fn with_row_lineage(mut self, probes: RowLineageProbes) -> Self {
        self.row_lineage = Some(Arc::new(probes));
        self
    }
}

impl PhysicalExprAdapterFactory for FieldIdExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> DataFusionResult<Arc<dyn PhysicalExprAdapter>> {
        let field_ids = arrow_schema_field_ids(&physical_file_schema);
        // A file without field ids (an external or pre-DuckLake parquet file)
        // has nothing to resolve by id, and is matched by name.
        if field_ids.is_empty() {
            return DuckLakeDefaultExprAdapterFactory
                .create(logical_file_schema, physical_file_schema);
        }

        // The scan's logical schema is the catalog columns, then any virtual
        // columns the scan asks the reader for (a row position), which the file
        // does not store and which need no resolving.
        let scan_schema = logical_file_schema;
        let logical_file_schema = if scan_schema.fields().len() > self.columns.len() {
            Arc::new(Schema::new(
                scan_schema.fields()[..self.columns.len()].to_vec(),
            ))
        } else {
            Arc::clone(&scan_schema)
        };

        // The read schema names each column as this file stores it, carries an
        // absent column's `initial_default`, and keeps the file's own nested
        // names (a list element the file calls `element`); `CatalogTypeExpr`
        // relabels those to the catalog's.
        let (read_schema, _renames) = build_read_schema_with_field_id_mapping_from_schema(
            &self.columns,
            &logical_file_schema,
            &field_ids,
            Some(&physical_file_schema),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let read_schema = Arc::new(read_schema);

        let inner = DuckLakeDefaultExprAdapterFactory
            .create(Arc::clone(&read_schema), Arc::clone(&physical_file_schema))?;
        Ok(Arc::new(FieldIdExprAdapter {
            columns: Arc::clone(&self.columns),
            row_lineage: self.row_lineage.clone(),
            scan_schema,
            logical_file_schema,
            read_schema,
            physical_file_schema,
            inner,
        }))
    }
}

/// Rewrites expressions over the catalog columns into expressions over one
/// file's columns.
///
/// Each catalog column is replaced by the file column that stores it: its field
/// of the per-file read schema, which sits at the same index. A file column whose
/// type is the read type is referenced as it is; any other is resolved by the
/// adapter every DuckLake scan uses, which casts it to the read type, or fills a
/// column the file does not carry. The column is then wrapped in a
/// [`CatalogTypeExpr`] where its read type holds different values from the
/// catalog's, and the whole expression where its result differs from the
/// catalog's only by nested field metadata.
///
/// A struct child access, `get_field(s, 'a', …)`, is resolved as a whole: each
/// child is looked up by field id in this file and read by the name the file stores
/// it under, from the bare file column. A child the file predates reads as its
/// `initial_default` where the struct holding it is valid and NULL where it is
/// not, with that validity read through the same `get_field` form (see
/// [`DefaultUnderParentExpr`]). That keeps it the one shape of struct access the
/// parquet reader's row filter accepts. DataFusion has by then removed the
/// `FilterExec` above the scan for any predicate that shape allows, so a predicate
/// the row filter declined would be lost rather than re-applied.
#[derive(Debug)]
struct FieldIdExprAdapter {
    columns: Arc<[DuckLakeTableColumn]>,
    row_lineage: Option<Arc<RowLineageProbes>>,
    /// The scan's logical schema: the catalog columns, then its virtual columns.
    scan_schema: SchemaRef,
    /// The catalog columns alone.
    logical_file_schema: SchemaRef,
    read_schema: SchemaRef,
    physical_file_schema: SchemaRef,
    inner: Arc<dyn PhysicalExprAdapter>,
}

impl FieldIdExprAdapter {
    /// The expression reading catalog column `index` from this file, typed as
    /// the read schema's field.
    fn read_column(&self, index: usize) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        if let Some(column) = self.bare_file_column(index) {
            // Referenced bare, not through a cast that only restates field
            // metadata or nullability, so the row filter can accept it.
            return Ok(column);
        }
        self.inner.rewrite(Arc::new(Column::new(
            self.read_schema.field(index).name(),
            index,
        )))
    }

    /// Catalog column `index` as the file column it is stored in, when that
    /// column already has the read type.
    fn bare_file_column(&self, index: usize) -> Option<Arc<dyn PhysicalExpr>> {
        let read_field = self.read_schema.field(index);
        let physical_index = self.physical_file_schema.index_of(read_field.name()).ok()?;
        (self.physical_file_schema.field(physical_index).data_type() == read_field.data_type())
            .then(|| Arc::new(Column::new(read_field.name(), physical_index)) as _)
    }

    /// Catalog column `index`, read as the catalog's type.
    fn catalog_column(&self, index: usize) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        let catalog_field = self.logical_file_schema.field(index);
        let read_field = self.read_schema.field(index);
        let read_column = self.read_column(index)?;
        // A read type that differs from the catalog's only by nested field
        // metadata — the field ids a file tags each nested node with — holds the
        // same values, and stays bare. Its relabel is applied to the result.
        if types_equal_ignoring_field_metadata(read_field.data_type(), catalog_field.data_type()) {
            return Ok(read_column);
        }
        Ok(Arc::new(CatalogTypeExpr::with_defaults(
            read_column,
            Arc::new(catalog_field.clone()),
            DefaultFill::build(
                read_field.data_type(),
                catalog_field.data_type(),
                &self.columns[index],
            )?,
        )))
    }

    /// `get_field(column, path…)` over a struct catalog column, resolved in this
    /// file, or `None` when it is not one: the root is not a catalog column, a path
    /// element is not a string literal, or the file lacks the whole column (read
    /// like any other column the file does not carry).
    fn struct_access(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> DataFusionResult<Option<Arc<dyn PhysicalExpr>>> {
        let Some((get_field, column, path)) = struct_access_chain(expr) else {
            return Ok(None);
        };
        let Ok(index) = self.logical_file_schema.index_of(column.name()) else {
            return Ok(None);
        };
        if !matches!(
            self.logical_file_schema.field(index).data_type(),
            DataType::Struct(_)
        ) {
            return Ok(None);
        }
        let read_field = self.read_schema.field(index);
        let Ok(physical_index) = self.physical_file_schema.index_of(read_field.name()) else {
            return Ok(None);
        };
        let Ok(catalog_result) = expr.return_field(&self.scan_schema) else {
            return Ok(None);
        };
        let root: Arc<dyn PhysicalExpr> = Arc::new(Column::new(read_field.name(), physical_index));

        // Walk the path through the catalog type and the read type side by side.
        // The read type lists a struct's children in catalog order, each under the
        // name this file stores it with, or under an absent-field name. A map
        // lookup's key is a value, not a name, and passes through unchanged.
        let mut catalog_type = self.logical_file_schema.field(index).data_type();
        let mut read_type = read_field.data_type();
        let mut file_path: Vec<Arc<dyn PhysicalExpr>> = Vec::with_capacity(path.len());
        for (step, (key, key_expr)) in path.iter().enumerate() {
            match (catalog_type, read_type) {
                (DataType::Struct(catalog_children), DataType::Struct(read_children)) => {
                    let Some(position) = catalog_children.iter().position(|c| c.name() == key)
                    else {
                        return Ok(None);
                    };
                    let read_child = &read_children[position];
                    if let Some(field_id) = read_child.name().strip_prefix(ABSENT_FIELD_PREFIX) {
                        return self
                            .absent_child(
                                get_field,
                                &root,
                                file_path,
                                field_id,
                                index,
                                step + 1 == path.len(),
                                &catalog_result,
                            )
                            .map(Some);
                    }
                    file_path.push(Arc::new(Literal::new(ScalarValue::Utf8(Some(
                        read_child.name().clone(),
                    )))));
                    catalog_type = catalog_children[position].data_type();
                    read_type = read_child.data_type();
                },
                (DataType::Map(catalog_entries, _), DataType::Map(read_entries, _)) => {
                    file_path.push(Arc::clone(key_expr));
                    let (Some(catalog_value), Some(read_value)) = (
                        map_value_type(catalog_entries.data_type()),
                        map_value_type(read_entries.data_type()),
                    ) else {
                        return Ok(None);
                    };
                    catalog_type = catalog_value;
                    read_type = read_value;
                },
                _ => return Ok(None),
            }
        }

        let mut args = vec![root];
        args.extend(file_path);
        let access = get_field_expr(get_field, args, &self.physical_file_schema)?;
        let file_result = access.return_field(&self.physical_file_schema)?;
        if file_result.data_type() == catalog_result.data_type() {
            return Ok(Some(access));
        }
        if !file_result.data_type().is_nested() {
            return Ok(Some(Arc::new(CastExpr::new(
                access,
                catalog_result.data_type().clone(),
                None,
            ))));
        }
        Ok(Some(Arc::new(CatalogTypeExpr::with_defaults(
            access,
            catalog_result,
            DefaultFill::build(read_type, catalog_type, &self.columns[index])?,
        ))))
    }

    /// A struct child this file predates, reached at `parent_path` under `root`:
    /// its `initial_default` wherever the struct holding it is valid, and NULL
    /// where that struct, or any struct above it, is NULL — as official DuckLake's
    /// `remap_struct` gives a defaulted child its parent's validity.
    #[allow(clippy::too_many_arguments)]
    fn absent_child(
        &self,
        get_field: &ScalarFunctionExpr,
        root: &Arc<dyn PhysicalExpr>,
        parent_path: Vec<Arc<dyn PhysicalExpr>>,
        field_id: &str,
        index: usize,
        is_leaf: bool,
        catalog_result: &FieldRef,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        let default = if is_leaf {
            nested_initial_default(&self.columns[index], field_id, catalog_result.data_type())?
        } else {
            // A path through the absent child reads into its default; a struct
            // default is NULL in every DuckLake catalog this reads.
            None
        };
        let Some(default) = default else {
            return Ok(Arc::new(Literal::new(ScalarValue::try_from(
                catalog_result.data_type(),
            )?)));
        };
        // The parent's validity comes from reading one of its present leaves,
        // through `get_field` over the bare file column: the one form of struct
        // access the parquet reader's row filter accepts.
        let parent_type = self.file_type_at(root, &parent_path)?;
        let Some((leaf_path, leaf_is_list)) = readable_leaf(&parent_type) else {
            return Err(DataFusionError::NotImplemented(format!(
                "reading struct field default of column '{}': the file's struct holding it \
                 has no primitive or list field to read its validity from",
                self.columns[index].column_name
            )));
        };
        let parent_depth = parent_path.len();
        let mut args = vec![Arc::clone(root)];
        args.extend(parent_path);
        args.extend(
            leaf_path
                .into_iter()
                .map(|name| Arc::new(Literal::new(ScalarValue::Utf8(Some(name)))) as _),
        );
        let mut probe = get_field_expr(get_field, args, &self.physical_file_schema)?;
        if leaf_is_list {
            // A list leaf is accepted in a row filter only under a list predicate.
            probe = Arc::new(IsNotNullExpr::new(probe));
        }
        Ok(Arc::new(DefaultUnderParentExpr {
            probe,
            parent_depth,
            value: default,
            target: Arc::clone(catalog_result),
        }))
    }

    /// A row-lineage probe column resolved in this file, or `None` when `name`
    /// is not one.
    fn row_lineage_probe(&self, name: &str) -> Option<Arc<dyn PhysicalExpr>> {
        let probes = self.row_lineage.as_ref()?;
        let tagged = |field_id: i32| {
            self.physical_file_schema.fields().iter().position(|field| {
                field
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .is_some_and(|id| *id == field_id.to_string())
            })
        };
        let int64_or_null = |index: Option<usize>| -> Arc<dyn PhysicalExpr> {
            match index {
                Some(index) => {
                    let field = self.physical_file_schema.field(index);
                    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new(field.name(), index));
                    if field.data_type() == &DataType::Int64 {
                        column
                    } else {
                        Arc::new(CastExpr::new(column, DataType::Int64, None))
                    }
                },
                None => Arc::new(Literal::new(ScalarValue::Int64(None))),
            }
        };
        if probes.has_embedded.as_deref() == Some(name) {
            return Some(Arc::new(Literal::new(ScalarValue::Boolean(Some(
                tagged(ROW_ID_PARQUET_FIELD_ID).is_some(),
            )))));
        }
        if probes.embedded.as_deref() == Some(name) {
            return Some(int64_or_null(tagged(ROW_ID_PARQUET_FIELD_ID)));
        }
        if probes.embedded_snapshot.as_deref() == Some(name) {
            return Some(int64_or_null(tagged(SNAPSHOT_ID_PARQUET_FIELD_ID)));
        }
        None
    }

    /// The file's type at `path` under `root`.
    fn file_type_at(
        &self,
        root: &Arc<dyn PhysicalExpr>,
        path: &[Arc<dyn PhysicalExpr>],
    ) -> DataFusionResult<DataType> {
        if path.is_empty() {
            return root.data_type(&self.physical_file_schema);
        }
        let template = ScalarFunctionExpr::try_new(
            Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::new())),
            std::iter::once(Arc::clone(root))
                .chain(path.iter().cloned())
                .collect(),
            &self.physical_file_schema,
            Arc::new(ConfigOptions::default()),
        )?;
        template.data_type(&self.physical_file_schema)
    }
}

/// The value type of a map whose entries are `entries`.
fn map_value_type(entries: &DataType) -> Option<&DataType> {
    match entries {
        DataType::Struct(fields) if fields.len() == 2 => Some(fields[1].data_type()),
        _ => None,
    }
}

/// A path from a struct of type `data_type` to a leaf the row filter can read: a
/// primitive leaf if it has one, else a list; and whether the leaf is a list.
fn readable_leaf(data_type: &DataType) -> Option<(Vec<String>, bool)> {
    fn find(data_type: &DataType, lists: bool) -> Option<Vec<String>> {
        let DataType::Struct(children) = data_type else {
            return None;
        };
        for child in children {
            let found = match child.data_type() {
                DataType::Struct(_) => find(child.data_type(), lists),
                DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
                    lists.then(Vec::new)
                },
                other if !other.is_nested() => Some(Vec::new()),
                _ => None,
            };
            if let Some(mut rest) = found {
                rest.insert(0, child.name().clone());
                return Some(rest);
            }
        }
        None
    }
    find(data_type, false)
        .map(|path| (path, false))
        .or_else(|| find(data_type, true).map(|path| (path, true)))
}

/// A `get_field` chain over a column: the outermost call, the root column, and
/// each key as a string beside the literal it came from.
type StructAccess<'a> = (
    &'a ScalarFunctionExpr,
    &'a Column,
    Vec<(String, Arc<dyn PhysicalExpr>)>,
);

/// The `get_field` call, its root column and its path, for a chain of `get_field`
/// calls over a column whose keys are all string literals: each key as a string
/// and as the literal it came from. `get_field(get_field(s, 'a'), 'b')` and
/// `get_field(s, 'a', 'b')` give the same path.
fn struct_access_chain(expr: &Arc<dyn PhysicalExpr>) -> Option<StructAccess<'_>> {
    let get_field = ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(expr.as_ref())?;
    let (root, keys) = get_field.args().split_first()?;
    let mut path = Vec::with_capacity(keys.len());
    for key in keys {
        let name = key
            .downcast_ref::<Literal>()?
            .value()
            .try_as_str()
            .flatten()?
            .to_string();
        path.push((name, Arc::clone(key)));
    }
    if let Some(column) = root.downcast_ref::<Column>() {
        return Some((get_field, column, path));
    }
    let (_, column, mut inner_path) = struct_access_chain(root)?;
    inner_path.append(&mut path);
    Some((get_field, column, inner_path))
}

/// A `get_field` call like `template`, over `args`, typed against `schema`.
fn get_field_expr(
    template: &ScalarFunctionExpr,
    args: Vec<Arc<dyn PhysicalExpr>>,
    schema: &Schema,
) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(template.fun().clone()),
        args,
        schema,
        Arc::new(template.config_options().clone()),
    )?))
}

/// The `initial_default` of the nested node `field_id` of `column`, decoded as
/// `data_type`; `None` when it has none. `NULL` is DuckLake's spelling of none.
fn nested_initial_default(
    column: &DuckLakeTableColumn,
    field_id: &str,
    data_type: &DataType,
) -> DataFusionResult<Option<ScalarValue>> {
    let Some(value) = field_id
        .parse::<i64>()
        .ok()
        .and_then(|id| column.nested_initial_defaults.get(&id))
        .filter(|value| !value.eq_ignore_ascii_case("NULL"))
    else {
        return Ok(None);
    };
    parse_ducklake_default_scalar(value, data_type)
        .map(Some)
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "Cannot decode initial_default '{value}' of a nested field of column '{}' as \
                 {data_type}",
                column.column_name
            ))
        })
}

impl PhysicalExprAdapter for FieldIdExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        let catalog_result = expr.return_field(&self.scan_schema).ok();
        // Top-down, and not into what it produces: a replacement already refers to
        // this file's columns, whose names can be another catalog column's.
        let per_file = Arc::clone(&expr)
            .transform_down(|expr| {
                if let Some(access) = self.struct_access(&expr)? {
                    return Ok(Transformed::new(access, true, TreeNodeRecursion::Jump));
                }
                let Some(column) = expr.downcast_ref::<Column>() else {
                    return Ok(Transformed::no(expr));
                };
                if let Some(resolved) = self.row_lineage_probe(column.name()) {
                    return Ok(Transformed::new(resolved, true, TreeNodeRecursion::Jump));
                }
                let Ok(index) = self.logical_file_schema.index_of(column.name()) else {
                    // A virtual column, which the reader appends after the
                    // file's own columns.
                    return match self.physical_file_schema.index_of(column.name()) {
                        Ok(physical_index) => Ok(Transformed::new(
                            Arc::new(Column::new(column.name(), physical_index)),
                            true,
                            TreeNodeRecursion::Jump,
                        )),
                        Err(_) => Ok(Transformed::no(expr)),
                    };
                };
                Ok(Transformed::new(
                    self.catalog_column(index)?,
                    true,
                    TreeNodeRecursion::Jump,
                ))
            })
            .data()?;
        // The result must carry the catalog's type exactly, nested metadata
        // included: the scan emits it under the catalog schema. A predicate is
        // boolean either way, so this never wraps one.
        Ok(
            match (
                catalog_result,
                per_file.return_field(&self.physical_file_schema).ok(),
            ) {
                (Some(catalog), Some(read)) if catalog.data_type() != read.data_type() => {
                    Arc::new(CatalogTypeExpr::new(per_file, catalog))
                },
                _ => per_file,
            },
        )
    }
}

/// Where a nested value read from a file takes a child's `initial_default`: the
/// struct children the file predates, at any depth.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefaultFill {
    /// Per struct child, by position: its default, or a fill inside it.
    Struct(Vec<(usize, ChildFill)>),
    /// A fill inside each list element.
    List(Box<DefaultFill>),
    /// A fill inside each map entry.
    Map(Box<DefaultFill>),
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) enum ChildFill {
    Value(ScalarValue),
    Nested(DefaultFill),
}

impl DefaultFill {
    /// The fill a value read as `read` needs to present `catalog`, where the read
    /// type names each child the file predates with an absent-field name.
    pub(crate) fn build(
        read: &DataType,
        catalog: &DataType,
        column: &DuckLakeTableColumn,
    ) -> DataFusionResult<Option<Self>> {
        Ok(match (read, catalog) {
            (DataType::Struct(read_children), DataType::Struct(catalog_children))
                if read_children.len() == catalog_children.len() =>
            {
                let mut fills = Vec::new();
                for (position, (read_child, catalog_child)) in
                    read_children.iter().zip(catalog_children).enumerate()
                {
                    if let Some(field_id) = read_child.name().strip_prefix(ABSENT_FIELD_PREFIX) {
                        if let Some(value) =
                            nested_initial_default(column, field_id, catalog_child.data_type())?
                        {
                            fills.push((position, ChildFill::Value(value)));
                        }
                    } else if let Some(fill) =
                        Self::build(read_child.data_type(), catalog_child.data_type(), column)?
                    {
                        fills.push((position, ChildFill::Nested(fill)));
                    }
                }
                (!fills.is_empty()).then_some(Self::Struct(fills))
            },
            (
                DataType::List(read_element)
                | DataType::LargeList(read_element)
                | DataType::FixedSizeList(read_element, _),
                DataType::List(catalog_element)
                | DataType::LargeList(catalog_element)
                | DataType::FixedSizeList(catalog_element, _),
            ) => Self::build(
                read_element.data_type(),
                catalog_element.data_type(),
                column,
            )?
            .map(|fill| Self::List(Box::new(fill))),
            (DataType::Map(read_entries, _), DataType::Map(catalog_entries, _)) => Self::build(
                read_entries.data_type(),
                catalog_entries.data_type(),
                column,
            )?
            .map(|fill| Self::Map(Box::new(fill))),
            _ => None,
        })
    }

    /// Apply the fill to `array`, already of the catalog's type.
    pub(crate) fn apply(&self, array: &ArrayRef) -> DataFusionResult<ArrayRef> {
        Ok(match self {
            Self::Struct(fills) => {
                let array = array.as_struct();
                let mut columns = array.columns().to_vec();
                for (position, fill) in fills {
                    columns[*position] = match fill {
                        // A defaulted child takes its struct's validity, as
                        // official's `remap_struct` gives it.
                        ChildFill::Value(value) => {
                            let values = value.to_array_of_size(array.len())?;
                            match array.logical_nulls() {
                                Some(validity) => with_nulls(&values, validity)?,
                                None => values,
                            }
                        },
                        ChildFill::Nested(fill) => fill.apply(&columns[*position])?,
                    };
                }
                Arc::new(StructArray::try_new(
                    array.fields().clone(),
                    columns,
                    array.nulls().cloned(),
                )?)
            },
            Self::List(fill) => match array.data_type() {
                DataType::List(field) => {
                    let array = array.as_list::<i32>();
                    Arc::new(ListArray::try_new(
                        Arc::clone(field),
                        array.offsets().clone(),
                        fill.apply(array.values())?,
                        array.nulls().cloned(),
                    )?)
                },
                DataType::LargeList(field) => {
                    let array = array.as_list::<i64>();
                    Arc::new(LargeListArray::try_new(
                        Arc::clone(field),
                        array.offsets().clone(),
                        fill.apply(array.values())?,
                        array.nulls().cloned(),
                    )?)
                },
                DataType::FixedSizeList(field, size) => {
                    let array = array.as_fixed_size_list();
                    Arc::new(FixedSizeListArray::try_new(
                        Arc::clone(field),
                        *size,
                        fill.apply(array.values())?,
                        array.nulls().cloned(),
                    )?)
                },
                other => {
                    return Err(DataFusionError::Internal(format!(
                        "list default fill applied to {other}"
                    )));
                },
            },
            Self::Map(fill) => {
                let DataType::Map(field, ordered) = array.data_type() else {
                    return Err(DataFusionError::Internal(format!(
                        "map default fill applied to {}",
                        array.data_type()
                    )));
                };
                let array = array.as_map();
                let entries: ArrayRef = Arc::new(array.entries().clone());
                Arc::new(MapArray::try_new(
                    Arc::clone(field),
                    array.offsets().clone(),
                    fill.apply(&entries)?.as_struct().clone(),
                    array.nulls().cloned(),
                    *ordered,
                )?)
            },
        })
    }
}

/// A struct child's default, NULL wherever the struct holding it is NULL.
///
/// `probe` is `get_field(file column, parent path…, leaf…)`, possibly under
/// `IS NOT NULL`; its first `parent_depth` keys lead to the struct holding the
/// child. Only `probe`'s own column is ever read, so the parquet reader's row filter
/// sees exactly the struct access it accepts, and reads the leaf along with the
/// validity of every struct on the way to it.
#[derive(Debug, Eq)]
struct DefaultUnderParentExpr {
    probe: Arc<dyn PhysicalExpr>,
    parent_depth: usize,
    value: ScalarValue,
    target: FieldRef,
}

impl PartialEq for DefaultUnderParentExpr {
    fn eq(&self, other: &Self) -> bool {
        self.probe.eq(&other.probe)
            && self.parent_depth == other.parent_depth
            && self.value == other.value
            && self.target == other.target
    }
}

impl Hash for DefaultUnderParentExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.probe.hash(state);
        self.parent_depth.hash(state);
        self.value.hash(state);
        self.target.hash(state);
    }
}

impl DefaultUnderParentExpr {
    /// The `get_field` call inside `probe`.
    fn access(&self) -> DataFusionResult<&ScalarFunctionExpr> {
        let probe = match self.probe.downcast_ref::<IsNotNullExpr>() {
            Some(is_not_null) => is_not_null.arg(),
            None => &self.probe,
        };
        ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(probe.as_ref()).ok_or_else(|| {
            DataFusionError::Internal("struct field default probe is not get_field".into())
        })
    }

    /// Rows where the struct holding the child, and every struct above it, is
    /// valid.
    fn parent_validity(&self, batch: &RecordBatch) -> DataFusionResult<Option<NullBuffer>> {
        let access = self.access()?;
        let args = access.args();
        let mut validity: Option<NullBuffer> = None;
        for depth in 0..=self.parent_depth {
            let level = if depth == 0 {
                Arc::clone(&args[0])
            } else {
                Arc::new(ScalarFunctionExpr::try_new(
                    Arc::new(access.fun().clone()),
                    args[..=depth].to_vec(),
                    batch.schema_ref(),
                    Arc::new(access.config_options().clone()),
                )?) as Arc<dyn PhysicalExpr>
            };
            let level = level.evaluate(batch)?.into_array(batch.num_rows())?;
            validity = NullBuffer::union(validity.as_ref(), level.logical_nulls().as_ref());
        }
        Ok(validity)
    }
}

impl fmt::Display for DefaultUnderParentExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ducklake_field_default({}, {}, {})",
            self.probe, self.parent_depth, self.value
        )
    }
}

impl PhysicalExpr for DefaultUnderParentExpr {
    fn return_field(&self, _input_schema: &Schema) -> DataFusionResult<FieldRef> {
        Ok(Arc::new(self.target.as_ref().clone().with_nullable(true)))
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        let values = self.value.to_array_of_size(batch.num_rows())?;
        let Some(validity) = self.parent_validity(batch)? else {
            return Ok(ColumnarValue::Array(values));
        };
        Ok(ColumnarValue::Array(with_nulls(&values, validity)?))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.probe]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "ducklake_field_default takes one child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self {
            probe: children.swap_remove(0),
            parent_depth: self.parent_depth,
            value: self.value.clone(),
            target: Arc::clone(&self.target),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// `values`, which has no nulls, NULL wherever `validity` is.
fn with_nulls(values: &ArrayRef, validity: NullBuffer) -> DataFusionResult<ArrayRef> {
    Ok(make_array(
        values
            .to_data()
            .into_builder()
            .nulls(Some(validity))
            .build()?,
    ))
}

/// Presents a column read under a file's own layout as the catalog's type.
///
/// The value conversion is [`coerce_column`]'s: a cast where the value types
/// differ (a DuckDB `ARRAY` read as `FixedSizeList` for a catalog `List`), then a
/// relabel of nested field names and metadata to the catalog's (a struct child the
/// file stores under an older name, a list element named otherwise).
#[derive(Debug, Eq)]
pub(crate) struct CatalogTypeExpr {
    input: Arc<dyn PhysicalExpr>,
    target: FieldRef,
    /// Struct children this file predates that take a default.
    defaults: Option<Arc<DefaultFill>>,
}

// Written out rather than derived: deriving through `Arc<dyn PhysicalExpr>`
// hits rust-lang/rust#78808, as DataFusion's own `CastExpr` notes.
impl PartialEq for CatalogTypeExpr {
    fn eq(&self, other: &Self) -> bool {
        self.input.eq(&other.input)
            && self.target.eq(&other.target)
            && self.defaults == other.defaults
    }
}

impl Hash for CatalogTypeExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.target.hash(state);
        self.defaults.hash(state);
    }
}

impl CatalogTypeExpr {
    pub(crate) fn new(input: Arc<dyn PhysicalExpr>, target: FieldRef) -> Self {
        Self::with_defaults(input, target, None)
    }

    fn with_defaults(
        input: Arc<dyn PhysicalExpr>,
        target: FieldRef,
        defaults: Option<DefaultFill>,
    ) -> Self {
        Self {
            input,
            target,
            defaults: defaults.map(Arc::new),
        }
    }

    fn present(&self, array: &ArrayRef) -> DataFusionResult<ArrayRef> {
        let array = coerce_column(array, self.target.data_type())?;
        match &self.defaults {
            Some(defaults) => defaults.apply(&array),
            None => Ok(array),
        }
    }
}

impl fmt::Display for CatalogTypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ducklake_catalog_type({}, {})",
            self.input,
            self.target.data_type()
        )
    }
}

impl PhysicalExpr for CatalogTypeExpr {
    fn return_field(&self, _input_schema: &Schema) -> DataFusionResult<FieldRef> {
        Ok(Arc::clone(&self.target))
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        match self.input.evaluate(batch)? {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(self.present(&array)?)),
            ColumnarValue::Scalar(scalar) => {
                let array = self.present(&scalar.to_array_of_size(1)?)?;
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &array, 0,
                )?))
            },
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "ducklake_catalog_type takes one child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self {
            input: children.swap_remove(0),
            target: Arc::clone(&self.target),
            defaults: self.defaults.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ducklake_catalog_type(")?;
        self.input.fmt_sql(f)?;
        write!(f, ", {})", self.target.data_type())
    }
}

/// Whether every per-file rewrite of `predicate`, over a scan whose schema is
/// `schema`, is one the parquet reader's row filter accepts whenever it accepts
/// `predicate` itself.
///
/// The row filter accepts a struct column only as the root of a `get_field` over
/// the bare column, and [`FieldIdExprAdapter`] keeps a struct access in that form
/// only for a chain of string keys through structs to a leaf that is not itself a
/// struct or map. A predicate that reaches a struct or map column any other way
/// — bare, through a map lookup, with a key that is not a string literal — is
/// not guaranteed to be applied by the reader. Every other column is rewritten to
/// the file column, a cast of it, or a literal, which the row filter accepts
/// exactly when it accepts the original.
pub(crate) fn reader_keeps_predicate(predicate: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    let mut keeps = true;
    predicate
        .apply(|expr| {
            if let Some((_, column, path)) = struct_access_chain(expr)
                && let Ok(field) = schema.field_with_name(column.name())
                && matches!(field.data_type(), DataType::Struct(_))
            {
                let mut data_type = field.data_type();
                for (key, _) in &path {
                    let DataType::Struct(children) = data_type else {
                        keeps = false;
                        return Ok(TreeNodeRecursion::Stop);
                    };
                    let Some(child) = children.iter().find(|child| child.name() == key) else {
                        keeps = false;
                        return Ok(TreeNodeRecursion::Stop);
                    };
                    data_type = child.data_type();
                }
                if matches!(data_type, DataType::Struct(_) | DataType::Map(_, _)) {
                    keeps = false;
                    return Ok(TreeNodeRecursion::Stop);
                }
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Some(column) = expr.downcast_ref::<Column>()
                && schema.field_with_name(column.name()).is_ok_and(|field| {
                    matches!(field.data_type(), DataType::Struct(_) | DataType::Map(_, _))
                })
            {
                keeps = false;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("a predicate walk that never fails");
    keeps
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, lit};

    fn schema() -> Schema {
        let map = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let inner = DataType::Struct(vec![Field::new("p", DataType::Int32, true)].into());
        Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new(
                "s",
                DataType::Struct(
                    vec![
                        Field::new("x", DataType::Int32, true),
                        Field::new("n", inner, true),
                        Field::new("m", map.clone(), true),
                        Field::new("l", DataType::new_list(DataType::Int32, true), true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new("m", map.clone(), true),
            Field::new(
                "s2",
                DataType::Struct(vec![Field::new("q", DataType::Int32, true)].into()),
                true,
            ),
        ])
    }

    fn get_field(schema: &Schema, keys: &[ScalarValue]) -> Arc<dyn PhysicalExpr> {
        get_field_over(schema, "s", 1, keys)
    }

    fn get_field_over(
        schema: &Schema,
        column: &str,
        index: usize,
        keys: &[ScalarValue],
    ) -> Arc<dyn PhysicalExpr> {
        let mut args: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new(column, index))];
        args.extend(
            keys.iter()
                .map(|key| Arc::new(Literal::new(key.clone())) as _),
        );
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::new())),
                args,
                schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    fn equals(left: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, Operator::Eq, lit(1)))
    }

    fn null_check(expr: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(datafusion::physical_expr::expressions::IsNullExpr::new(
            expr,
        ))
    }

    fn key(name: &str) -> ScalarValue {
        ScalarValue::Utf8(Some(name.to_string()))
    }

    #[test]
    fn the_reader_keeps_a_primitive_column_and_a_struct_path_to_a_primitive_leaf() {
        let schema = schema();
        assert!(reader_keeps_predicate(
            &equals(Arc::new(Column::new("id", 0))),
            &schema
        ));
        assert!(reader_keeps_predicate(
            &equals(get_field(&schema, &[key("x")])),
            &schema
        ));
        assert!(reader_keeps_predicate(
            &equals(get_field(&schema, &[key("n"), key("p")])),
            &schema
        ));
        let list_leaf = get_field(&schema, &[key("l")]);
        assert!(reader_keeps_predicate(
            &(Arc::new(datafusion::physical_expr::expressions::IsNotNullExpr::new(
                list_leaf
            )) as Arc<dyn PhysicalExpr>),
            &schema
        ));
        assert!(reader_keeps_predicate(
            &(Arc::new(BinaryExpr::new(
                equals(get_field(&schema, &[key("x")])),
                Operator::And,
                equals(get_field_over(&schema, "s2", 3, &[key("q")])),
            )) as Arc<dyn PhysicalExpr>),
            &schema
        ));
    }

    #[test]
    fn the_reader_may_skip_a_bare_struct_a_struct_leaf_or_a_map_path() {
        let schema = schema();
        assert!(!reader_keeps_predicate(
            &null_check(Arc::new(Column::new("s", 1))),
            &schema
        ));
        assert!(!reader_keeps_predicate(
            &null_check(get_field(&schema, &[key("n")])),
            &schema
        ));
        assert!(!reader_keeps_predicate(
            &equals(get_field(&schema, &[key("m"), key("k")])),
            &schema
        ));
        assert!(!reader_keeps_predicate(
            &null_check(Arc::new(Column::new("m", 2))),
            &schema
        ));
        // A key that is not a literal, which only the planner's type check stops.
        let non_literal_key: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "get_field",
            Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::new())),
            vec![Arc::new(Column::new("s", 1)), Arc::new(Column::new("id", 0))],
            Arc::new(Field::new("x", DataType::Int32, true)),
            Arc::new(ConfigOptions::default()),
        ));
        assert!(!reader_keeps_predicate(&equals(non_literal_key), &schema));
        assert!(!reader_keeps_predicate(
            &(Arc::new(BinaryExpr::new(
                equals(get_field(&schema, &[key("x")])),
                Operator::And,
                null_check(Arc::new(Column::new("s2", 3))),
            )) as Arc<dyn PhysicalExpr>),
            &schema
        ));
    }
}
