//! DuckLake VARIANT access and file-statistics pruning.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use datafusion::common::{Result, ScalarValue, exec_err};
use datafusion::logical_expr::{
    ColumnarValue, Expr, Operator, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use parquet::variant::{GetOptions, VariantPath, VariantPathElement, variant_get};

use crate::metadata_provider::{DuckLakeFileVariantStatistics, DuckLakeTableColumn};
use crate::types::variant_data_type;

pub(crate) const VARIANT_GET_INT64_NAME: &str = "ducklake_variant_get_int64";

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct VariantGetInt64 {
    signature: Signature,
}

impl VariantGetInt64 {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![variant_data_type(), DataType::Utf8],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for VariantGetInt64 {
    fn name(&self) -> &str {
        VARIANT_GET_INT64_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let path = match args.args.get(1) {
            Some(ColumnarValue::Scalar(ScalarValue::Utf8(Some(path)))) => path.as_str(),
            _ => {
                return exec_err!(
                    "{} requires a non-null constant UTF-8 path",
                    VARIANT_GET_INT64_NAME
                );
            },
        };
        let path = VariantPath::try_from(path)?;
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let output = variant_get(
            &arrays[0],
            GetOptions::new_with_path(path).with_as_type(Some(Arc::new(Field::new(
                "value",
                DataType::Int64,
                true,
            )))),
        )?;
        Ok(ColumnarValue::Array(output))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantInt64Predicate {
    column_id: i64,
    variant_path: String,
    operator: Operator,
    value: i64,
}

impl VariantInt64Predicate {
    pub(crate) fn from_expr(expr: &Expr, columns: &[DuckLakeTableColumn]) -> Option<Self> {
        let Expr::BinaryExpr(binary) = expr else {
            return None;
        };
        if !matches!(
            binary.op,
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
        ) {
            return None;
        }

        if let Some((column_id, variant_path)) = variant_access(&binary.left, columns)
            && let Some(value) = int64_literal(&binary.right)
        {
            return Some(Self {
                column_id,
                variant_path,
                operator: binary.op,
                value,
            });
        }
        let (column_id, variant_path) = variant_access(&binary.right, columns)?;
        Some(Self {
            column_id,
            variant_path,
            operator: reverse_operator(binary.op)?,
            value: int64_literal(&binary.left)?,
        })
    }

    pub(crate) fn may_match(&self, statistics: &[DuckLakeFileVariantStatistics]) -> bool {
        let mut matches = statistics.iter().filter(|statistic| {
            statistic.column_id == self.column_id && statistic.variant_path == self.variant_path
        });
        let Some(statistic) = matches.next() else {
            return true;
        };
        if matches.next().is_some() || statistic.shredded_type != "int64" {
            return true;
        }
        let (Some(minimum), Some(maximum)) = (&statistic.min_value, &statistic.max_value) else {
            return true;
        };
        let (Ok(minimum), Ok(maximum)) = (minimum.parse::<i64>(), maximum.parse::<i64>()) else {
            return true;
        };

        if minimum > maximum {
            return true;
        }
        match self.operator {
            Operator::Eq => minimum <= self.value && self.value <= maximum,
            Operator::NotEq => minimum != self.value || maximum != self.value,
            Operator::Lt => minimum < self.value,
            Operator::LtEq => minimum <= self.value,
            Operator::Gt => maximum > self.value,
            Operator::GtEq => maximum >= self.value,
            _ => true,
        }
    }
}

fn variant_access(expr: &Expr, columns: &[DuckLakeTableColumn]) -> Option<(i64, String)> {
    let Expr::ScalarFunction(function) = expr else {
        return None;
    };
    if function.name() != VARIANT_GET_INT64_NAME {
        return None;
    }
    let [Expr::Column(column), Expr::Literal(ScalarValue::Utf8(Some(path)), _)] =
        function.args.as_slice()
    else {
        return None;
    };
    let column_id = columns
        .iter()
        .find(|candidate| candidate.column_name == column.name)?
        .column_id;
    Some((column_id, catalog_variant_path(path)?))
}

fn catalog_variant_path(path: &str) -> Option<String> {
    let path = VariantPath::try_from(path).ok()?;
    path.path()
        .iter()
        .map(|element| match element {
            VariantPathElement::Field {
                name,
            } => Some(format!("\"{}\"", name.replace('"', "\"\""))),
            VariantPathElement::Index {
                ..
            } => None,
        })
        .collect::<Option<Vec<_>>>()
        .map(|elements| elements.join("."))
}

fn int64_literal(expr: &Expr) -> Option<i64> {
    let Expr::Literal(value, _) = expr else {
        return None;
    };
    match value {
        ScalarValue::Int8(Some(value)) => Some(i64::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i64::from(*value)),
        ScalarValue::Int32(Some(value)) => Some(i64::from(*value)),
        ScalarValue::Int64(Some(value)) => Some(*value),
        ScalarValue::UInt8(Some(value)) => Some(i64::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i64::from(*value)),
        ScalarValue::UInt32(Some(value)) => Some(i64::from(*value)),
        ScalarValue::UInt64(Some(value)) => i64::try_from(*value).ok(),
        _ => None,
    }
}

fn reverse_operator(operator: Operator) -> Option<Operator> {
    match operator {
        Operator::Eq => Some(Operator::Eq),
        Operator::NotEq => Some(Operator::NotEq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{ScalarUDF, col, lit};

    fn column() -> DuckLakeTableColumn {
        DuckLakeTableColumn::new(17, "payload".to_string(), "variant".to_string(), true)
    }

    fn statistic(minimum: Option<&str>, maximum: Option<&str>) -> DuckLakeFileVariantStatistics {
        DuckLakeFileVariantStatistics {
            data_file_id: 9,
            column_id: 17,
            variant_path: "\"nested\".\"count\"".to_string(),
            shredded_type: "int64".to_string(),
            value_count: Some(10),
            null_count: Some(0),
            min_value: minimum.map(str::to_string),
            max_value: maximum.map(str::to_string),
            contains_nan: None,
        }
    }

    #[test]
    fn recognizes_int64_variant_predicate_and_prunes_disjoint_bounds() {
        let udf = ScalarUDF::new_from_impl(VariantGetInt64::new());
        let expression = udf
            .call(vec![col("payload"), lit("nested.count")])
            .gt_eq(lit(20_i64));
        let predicate = VariantInt64Predicate::from_expr(&expression, &[column()]).unwrap();

        assert_eq!(predicate.column_id, 17);
        assert_eq!(predicate.variant_path, "\"nested\".\"count\"");
        assert_eq!(predicate.operator, Operator::GtEq);
        assert_eq!(predicate.value, 20);
        assert!(!predicate.may_match(&[statistic(Some("0"), Some("19"))]));
        assert!(predicate.may_match(&[statistic(Some("0"), Some("20"))]));
        assert!(predicate.may_match(&[statistic(None, Some("19"))]));
        assert!(predicate.may_match(&[]));
    }
}
