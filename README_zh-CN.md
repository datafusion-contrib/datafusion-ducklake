# DataFusion-DuckLake

<!-- hy-mt2-i18n:start -->
[English](./README.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


[![crates.io](https://img.shields.io/crates/v/datafusion-ducklake.svg)](https://crates.io/crates/datafusion-ducklake)
[![docs.rs](https://img.shields.io/docsrs/datafusion-ducklake)](https://docs.rs/datafusion-ducklake)
[![CI](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml/badge.svg)](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-DataFusion%2BDuckLake-5865F2?logo=discord&logoColor=white)](https://discord.com/channels/885562378132000778/1492192627666321452)

一款用于读取和写入[DuckLake](https://ducklake.select)目录的[DataFusion](https://datafusion.apache.org/)扩展。DuckLake是一种集成的数据湖与目录格式，它将元数据存储在SQL数据库中，而数据则以Parquet文件的形式保存在磁盘或对象存储中。

该项目的目标是让 DuckLake 成为 DataFusion 中一流、原生支持 Arrow 的湖仓格式。

该项目由 [Hotdata](https://www.hotdata.dev) 维护，并得到了社区的支持。欢迎在 [Hotdata Discord](https://discord.gg/cdHczfxxBc) 上与我们交流。

- 📦 **crates.io:** <https://crates.io/crates/datafusion-ducklake>
- 📖 **API 文档:** <https://docs.rs/datafusion-ducklake>
- 🧩 **功能与后端支持:** 见 [COMPATIBILITY.md](COMPATIBILITY.md)
- 💬 **项目交流频道:** [DataFusion+DuckLake Discord](https://discord.com/channels/885562378132000778/1492192627666321452) — 用于讨论开发与使用相关问题
- 🧡 **团队介绍:** [Hotdata Discord](https://discord.gg/cdHczfxxBc)

# 快速入门

添加该 crate：

```bash
cargo add datafusion-ducklake
```

默认构建会静态绑定 DuckDB 目录后端。其他后端及写入支持需通过特性标志启用——完整兼容性矩阵请参见 [COMPATIBILITY.md](COMPATIBILITY.md)。

```toml
# Cargo.toml — 读取 PostgreSQL 目录
```

## 快速入门

添加该 crate：

```bash
cargo add datafusion-ducklake
```

默认构建已静态打包了 DuckDB 目录后端。其他后端及写入支持需通过特性标志手动启用——完整的兼容性矩阵请参见 [COMPATIBILITY.md](COMPATIBILITY.md)。

```toml
# Cargo.toml — 读取 PostgreSQL 目录
# （若要使用实验性的多目录写入功能，请设置 features = ["write-postgres"]）
[dependencies]
datafusion-ducklake = { version = "0.5", features = ["metadata-postgres"] }
```

下面的示例也直接使用了 `datafusion`、`object_store` 和 `url` —— 请将其一并添加到您的 `[dependencies]` 中（该 crate 不会重新导出这些模块）。写入示例还会使用 `sqlx`（并启用其 `postgres` 和 `runtime-tokio` 特性）来打开连接池。

使用附带的示例针对现有的 PostgreSQL 数据目录运行查询：

```bash
cargo run --example basic_query --features metadata-postgres -- \
  "postgresql://user:password@localhost:5432/database" "SELECT * FROM main.users"
```

（该示例在启用相应的 `metadata-*` 特性后，也支持 DuckDB、SQLite 和 MySQL 的连接字符串——详情请参阅 [COMPATIBILITY.md](COMPATIBILITY.md)。）

## 读取目录

## 读取目录

将 `DuckLakeCatalog` 注册到 `SessionContext` 中，然后通过常规 SQL 语句以 `catalog.schema.table` 的格式对其进行查询：

```rust
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use url::Url;

// （在异步函数内部）
// 从 PostgreSQL 目录中读取元数据
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;

// 为任何非本地数据注册对象存储（S3 / MinIO）
let runtime = Arc::new(RuntimeEnv::default());
let s3: Arc<dyn ObjectStore> = Arc::new(
    AmazonS3Builder::new()
       .with_endpoint("http://localhost:9000") // MinIO 端点
       .with_bucket_name("ducklake-data")
       .with_access_key_id("minioadmin")
       .with_secret_access_key("minioadmin")
       .with_region("us-west-2") // MinIO 任意区域均可
       .with_allow_http(true)    // http:// 端点所需
       .build()?,
);
runtime.register_object_store(&Url::parse("s3://ducklake-data/")?, s3);

let catalog = DuckLakeCatalog::new(provider)?;
let ctx = SessionContext::new_with_config_rt(
    SessionConfig::new().with_default_catalog_and_schema("ducklake", "main"),
    runtime,
);
ctx.register_catalog("ducklake", Arc::new(catalog));

let df = ctx.sql("SELECT * FROM ducklake.main.my_table").await?;
df.show().await?;
```

## 编写目录

## 编写目录

PostgreSQL 有两种写入实现，均基于 `write-postgres` 功能：

- **`PostgresSingleCatalogMetadataWriter`** — 这是符合规范的**标准单目录结构**。其目录格式与 SQLite 和 MySQL 写入器相同，因此其他 DuckLake 实现（包括 DuckDB 的 `ducklake` 扩展）也能读取（并写入）该目录。SQL 的 `CREATE TABLE AS SELECT` 和 `INSERT INTO` 语句均可使用。**建议优先选用此选项。**
- **`PostgresMetadataWriter`** — 即 [专门章节](#multi-catalog-postgresql-experimental)中描述的**实验性多目录结构**，用于在单个数据库中存储多个目录。该写法特定于某个库，并非 DuckLake 规范的一部分，且不支持 CTAS 语句。

```rust,ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
};
use std::sync::Arc;

// 初始化标准的DuckLake表，并将目录指向数据存储根路径
let writer = PostgresSingleCatalogMetadataWriter::new_with_init(
    "postgresql://user:pass@localhost:5432/db",
).await?;
writer.set_data_path("/abs/path/to/data")?;

// 在这种模式下，CTAS和INSERT语句都能正常使用
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("CREATE TABLE ducklake.main.events AS SELECT 1 AS id").await?.collect().await?;
```

多目录路径则如下所示——先通过写入器 API 创建表（不使用 CTAS），然后再通过 SQL 补充数据。

```rust
ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MulticatalogManager, MulticatalogProvider,
    PostgresMetadataWriter, initialize_multicatalog_schema,
};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;

let pool = PgPoolOptions::new().connect("postgresql://user:pass@localhost:5432/db").await?;

// 一次性初始化多目录表结构，随后创建命名目录
initialize_multicatalog_schema(&pool).await?;
let catalog_id = MulticatalogManager::new(pool.clone()).create_catalog("my_catalog").await?;

// 通过表写入器写入第一批数据来创建表
let writer = Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), catalog_id).await?);
writer.set_data_path("/abs/path/to/data")?;
let object_store: Arc<dyn object_store::ObjectStore> =
    Arc::new(object_store::local::LocalFileSystem::new());
let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store)?;
table_writer.write_table("public", "events", &[batch]).await?; // `batch` 即为你的 RecordBatch

// 接下来通过 SQL 进行追加操作，再次通过 MulticatalogProvider 访问同一目录
let provider = MulticatalogProvider::with_pool(pool.clone(), "my_catalog").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer)?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("INSERT INTO ducklake.public.events VALUES (1, 'a')").await?.collect().await?;
ctx.sql("SELECT count(*) FROM ducklake.public.events").await?.show().await?;
```

写入器的输出参数是可配置的（包括 Parquet 压缩格式，以及根据行数和字节数确定的行组大小）。有关写入器选项的详细信息，请参阅 [`DuckLakeTableWriter`](https://docs.rs/datafusion-ducklake)。

目前，通过 `SqliteMetadataWriter`（功能名为 `write-sqlite`），已支持向**标准、单目录**的 DuckLake 存储库（符合规范的结构）写入数据，该功能适用于 **SQLite**，且 SQL 的 `CREATE TABLE AS SELECT` 与 `INSERT INTO` 语句均可正常使用。详情请参见 [`tests/it/sql_write_tests.rs`](tests/it/sql_write_tests.rs)。

# 分区功能

## 分区功能

通过一个或多个列对表进行分区（可选择通过转换实现），这样基于分区列进行过滤的查询就可以跳过整个文件：

// 在加载数据之前先声明分区方案，之后再像往常一样执行插入操作。
execute_ducklake_sql(
    &ctx,
    &catalog,
    "ALTER TABLE ducklake.main.sales SET PARTITIONED BY (region, year(sale_date))",
)
.await?!

写入操作会将拆分后的行按每个分区值分别写入一个 Parquet 文件；读取时则会自动过滤掉不匹配的文件。支持的转换函数有 `identity`（原始值）以及 `year`/`month`/`day`/`hour`；目前过滤功能仅适用于 `identity` 和 `year`（`month`/`day`/`hour` 会被记录下来，但暂不会用于跳过文件）。目前 **SQLite** 已支持带分区的写入操作，而所有后端都支持读取并过滤文件的功能。使用 `RESET PARTITIONED BY` 可关闭此功能。详情请参阅 [COMPATIBILITY.md](COMPATIBILITY.md) 以及 [`tests/it/partition_write_tests.rs`](tests/it/partition_write_tests.rs)。

## 严格约束
1. **结构锁定**：绝对保持原有的 Markdown 数据结构、缩进、标题层级、表格、链接、URL、徽章、代码块和行内代码完全不变。
2. **选择性翻译**：仅翻译面向用户展示的可见自然语言内容。
3. **禁止修改**：**严禁**翻译或更改代码标签、键名、变量占位符（如 {{var}}、${var}、%s、%d 等）、命令示例、文件路径、项目名、API 名、包名、模型名、标识符和代码符号；除非背景信息中已经给出对应译名。
4. 术语、风格、专有名词的译法要与所给背景信息保持一致。

【待翻译片段】
---

## 多目录功能（PostgreSQL，实验性功能）

单个 PostgreSQL 元数据存储可以托管**多个独立的 DuckLake 目录**——这非常适合多租户部署场景，或是需要在同一个数据库中管理众多逻辑湖仓的情况。

> ⚠️ **实验性功能且仅适用于特定库。** 这种多目录架构**不属于DuckLake规范**，目前上游也尚未支持或认可该方案。以这种方式创建的目录只能通过该库的`MulticatalogProvider`进行读取——它们**无法**与标准的单目录DuckLake存储相互替换。相关API以及磁盘/目录内的结构都可能发生变化。由于当前对PostgreSQL的写入支持依赖于此路径，因此请将其视为预览版本。

- 使用 `MulticatalogManager`（功能 `write-postgres`）来**创建和管理**目录：`initialize_multicatalog_schema` 用于初始化共享表，随后通过 `create_catalog` 创建目录，而 `drop_table_in_catalog` 则用于管理目录中的内容。
- 通过 `MulticatalogProvider::with_pool(pool, "name")`（功能 `multicatalog-postgres`）来**读取**特定目录，该方式与其他元数据提供者类似，可直接与 `DuckLakeCatalog` 集成使用。

请参阅[`examples/multicatalog_write.rs`](examples/multicatalog_write.rs)，了解从启动初始化 → 创建目录 → 写入数据 → 读取回数据的完整流程演示。

## 维护功能

## 维护功能

`maintenance` API 负责从 Rust 层面处理湖仓的维护工作：过期旧快照的清理、被替代文件的整理，以及回收孤立文件的资源。具体的入口函数受后端限制（即 `write-sqlite` / `write-postgres`）。而 `DROP TABLE` 操作则可通过 `MetadataWriter` 来实现。详情请参阅
[`examples/maintenance_demo.rs`](examples/maintenance_demo.rs)
和
[`examples/orphan_cleanup_demo.rs`](examples/orphan_cleanup_demo.rs)。

### 压缩整理

针对 `DuckLakeTable` 的两种显式触发操作，可在不改变其逻辑行的前提下，将表的数据文件重写为更优的物理结构。

- `merge_adjacent_files(state, MergeOptions)` 会将多个小型文件（具有相同的模式版本）合并为数量更少但体积更大的文件。对于跨越多个原始快照的合并文件，会将其作为 DuckLake *部分数据文件* 写入（同时保留每行原有的 rowid 和原始快照），从而不会影响时间回溯功能及变更推送机制。
- `rewrite_data_files(state, RewriteOptions)` 会重写那些已删除比例超过阈值（默认为 `0.95`）的文件，即删除其中的已删除行。

这两项操作会在同一个快照中原子性地完成提交，并能与并发追加操作共存；被取代的文件会被安排删除，随后由 `cleanup_old_files` 函数进行回收。详见 [`examples/compaction_demo.rs`](examples/compaction_demo.rs)。

## 兼容性
如需了解目录后端、对象存储、类型、功能以及当前限制的完整说明，请参阅 **[COMPATIBILITY.md](COMPATIBILITY.md)**。
以下是几项需要提前了解的重点：
- 读取操作支持 DuckDB、SQLite、PostgreSQL 和 MySQL；**写入操作仅支持 SQLite/PostgreSQL**。
- 对象存储：本地文件系统及兼容 S3 的存储类型（S3、MinIO）。

## 兼容性

如需了解目录后端、对象存储、数据类型、功能以及当前限制的完整详情，请参阅**[COMPATIBILITY.md](COMPATIBILITY.md)**。

先介绍几个值得了解的重点：

- 读取操作支持 DuckDB、SQLite、PostgreSQL 和 MySQL；**写入操作仅支持 SQLite/PostgreSQL**。
- 对象存储：本地文件系统以及兼容 S3 的存储（S3、MinIO）。
- 可通过 `DuckLakeCatalog` 或在查询时使用 `ducklake_table_at` 来选择快照；DataFusion 不支持 `AS OF` 语法。
- 表分区：所有后端均支持读取操作及文件剪枝功能；SQLite 支持分区写入。
- 由 DuckDB 的 ducklake 扩展内联的数据**无法被读取**——有关 `COUNT(*)` 计数偏小的问题及规避方法，请参阅 COMPATIBILITY.md。

---

## 项目状态

该项目目前仍处于测试阶段，会随 DataFusion 和 DuckLake 的发展不断演进。随着核心抽象的完善，相关 API 可能会发生变化。有关版本历史记录，请参阅 [CHANGELOG.md](CHANGELOG.md)。我们欢迎任何反馈、问题报告以及贡献。
