# DataFusion-DuckLake

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | **日本語** | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


[![crates.io](https://img.shields.io/crates/v/datafusion-ducklake.svg)](https://crates.io/crates/datafusion-ducklake)
[![docs.rs](https://img.shields.io/docsrs/datafusion-ducklake)](https://docs.rs/datafusion-ducklake)
[![CI](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml/badge.svg)](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-DataFusion%2BDuckLake-5865F2?logo=discord&logoColor=white)](https://discord.com/channels/885562378132000778/1492192627666321452)

[DuckLake](https://ducklake.select)のカタログを読み書きするための[DataFusion](https://datafusion.apache.org/)拡張機能です。DuckLakeとは、メタデータをSQLデータベースに、データをディスクやオブジェクトストレージ上のParquetファイルとして保存する、統合型のデータレイクおよびカタログ形式です。

このプロジェクトの目標は、DataFusion内でDuckLakeを一流の、Arrowネイティブなレイクハウス形式として実現することです。

このプロジェクトはコミュニティのサポートを受けながら、[Hotdata](https://www.hotdata.dev)によってメンテナンスされています。開発や利用に関するご質問は、[HotdataのDiscordチャンネル](https://discord.gg/cdHczfxxBc)でお気軽にお声がけください。

- 📦 **crates.io:** <https://crates.io/crates/datafusion-ducklake>  
- 📖 **APIドキュメント:** <https://docs.rs/datafusion-ducklake>  
- 🧩 **機能およびバックエンドのサポート状況:** [COMPATIBILITY.md](COMPATIBILITY.md) をご覧ください  
- 💬 **プロジェクトのチャット:** [DataFusion+DuckLake Discord](https://discord.com/channels/885562378132000778/1492192627666321452) — 開発や利用に関する議論  
- 🧡 **チーム紹介:** [Hotdata Discord](https://discord.gg/cdHczfxxBc)

---

## すぐに始める

クレートを追加します：

```bash
cargo add datafusion-ducklake
```

デフォルトのビルドにはDuckDBカタログバックエンドが静的に組み込まれています。その他のバックエンドや書き込みサポートはフィーチャーフラグを通じてオプションで有効化でき、全ての互換性情報は[COMPATIBILITY.md](COMPATIBILITY.md)をご覧ください。

```toml
# Cargo.toml — PostgreSQLカタログの読み込み
# （実験的なマルチカタログ書き込み機能を使用する場合は、features = ["write-postgres"] を指定してください）
[dependencies]
datafusion-ducklake = { version = "0.5", features = ["metadata-postgres"] }
```

以下の例では `datafusion`、`object_store`、`url` も直接使用されているため、これらも `[dependencies]` に追加してください（このクレートはこれらを再エクスポートしません）。書き込み用の例では、接続プールを開くために `sqlx`（`postgres` および `runtime-tokio` フィーチャーを含む）も使用されています。

同梱されている例を使って、既存の PostgreSQL カタログに対してクエリを実行します：

```bash
cargo run --example basic_query --features metadata-postgres -- \
  "postgresql://user:password@localhost:5432/database" "SELECT * FROM main.users"
```

（この例では、対応する `metadata-*` フィーチャーを使用すれば、DuckDB、SQLite、MySQL の接続文字列も受け付けます——[COMPATIBILITY.md](COMPATIBILITY.md) を参照してください。）

# カタログの読み込み

## カタログの読み込み

`SessionContext`に`DuckLakeCatalog`を登録し、通常のSQLを使って`catalog.schema.table`としてクエリを実行します：

```rust
ignore
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use url::Url;

// (async fn の内部で)
// PostgreSQL カタログからメタデータを読み込む
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;

// ローカル外のデータ用にオブジェクトストアを登録する（S3 / MinIO）
let runtime = Arc::new(RuntimeEnv::default());
let s3: Arc<dyn ObjectStore> = Arc::new(
    AmazonS3Builder::new()
       .with_endpoint("http://localhost:9000") // MinIO のエンドポイント
       .with_bucket_name("ducklake-data")
       .with_access_key_id("minioadmin")
       .with_secret_access_key("minioadmin")
       .with_region("us-west-2") // MinIO ではどのリージョンでも可
       .with_allow_http(true)    // http:// エンドポイントに必要
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

# 厳格な制約事項
1. **構造の維持**：元の Markdown のデータ構造、インデント、見出し階層、表、リンク、URL、バッジ、コードブロック、およびインラインコードを一切変更しないこと。
2. **選択的翻訳**：ユーザーに表示される可視的な自然言語テキストのみを翻訳すること。
3. **変更禁止**：コードのラベル、キー名、変数プレースホルダー（{{var}}、${var}、%s、%d など）、コマンド例、ファイルパス、プロジェクト名、API名、パッケージ名、モデル名、識別子、コード記号を翻訳したり変更したりすることは**厳禁**である。ただし、背景情報に既に対応する訳名が記載されている場合は除く。
4. 用語、スタイル、固有名詞の翻訳は、提供された背景情報と一致させること。

## カタログの書き込み

PostgreSQLには2つのライターがあり、どちらも`write-postgres`機能を介して利用されます：

- **`PostgresSingleCatalogMetadataWriter`** — 標準で仕様に準拠した
  シングルカタログ形式です。SQLiteやMySQLのライターと同じカタログ構造を持つため、DuckDBの`ducklake`拡張機能を含む他のDuckLake実装でもそのカタログを読み書きできます。SQLの`CREATE TABLE AS SELECT`や`INSERT INTO`も両方利用可能です。**こちらを優先してください。**
- **`PostgresMetadataWriter`** — [別のセクション](#multi-catalog-postgresql-experimental)で説明されている
  実験的なマルチカタログ形式で、1つのデータベース内に複数のカタログを格納するためのものです。これは特定のライブラリ専用でDuckLakeの仕様には含まれておらず、CTASもサポートしていません。

```rust,ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
};
use std::sync::Arc;

// 標準のDuckLakeテーブルを初期化し、カタログをデータのルートに設定する
let writer = PostgresSingleCatalogMetadataWriter::new_with_init(
    "postgresql://user:pass@localhost:5432/db",
).await?;
writer.set_data_path("/abs/path/to/data")?;

// このパスではCTASとINSERTの両方が利用可能だ
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("CREATE TABLE ducklake.main.events AS SELECT 1 AS id").await?.collect().await?;
```

マルチカタログのパスはこのようになります。テーブルはwriter APIを通じて作成され（CTASは使用しない）、その後SQLによって追加されます。

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

// 一度だけ実行：マルチカタログ用のテーブルを初期化し、名前付きカタログを作成する
initialize_multicatalog_schema(&pool).await?;
let catalog_id = MulticatalogManager::new(pool.clone()).create_catalog("my_catalog").await?;

// テーブルライターを通じて最初のバッチを書き込むことでテーブルを作成する
let writer = Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), catalog_id).await?);
writer.set_data_path("/abs/path/to/data")?;
let object_store: Arc<dyn object_store::ObjectStore> =
    Arc::new(object_store::local::LocalFileSystem::new());
let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store)?;
table_writer.write_table("public", "events", &[batch]).await?; // `batch`はRecordBatchです

// 次にSQLを使って追加処理を行い、MulticatalogProviderを通じて同じカタログを読み取る
let provider = MulticatalogProvider::with_pool(pool.clone(), "my_catalog").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer)?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("INSERT INTO ducklake.public.events VALUES (1, 'a')").await?.collect().await?;
ctx.sql("SELECT count(*) FROM ducklake.public.events").await?.show().await?;
```

Writerの出力は設定可能であり（Parquet圧縮、行数およびバイトサイズに基づく行グループのサイズ調整など）、詳細なオプションについては[`DuckLakeTableWriter`](https://docs.rs/datafusion-ducklake)を参照してください。

現在では、`SqliteMetadataWriter`（機能 `write-sqlite`）を通じて**SQLite**向けの**標準的な単一カタログ**構成（仕様に準拠したレイアウト）への書き込みがサポートされており、SQLの`CREATE TABLE AS SELECT`や`INSERT INTO`も両方利用可能です。詳細は[`tests/it/sql_write_tests.rs`](tests/it/sql_write_tests.rs)をご覧ください。

# 厳格な制約事項

## パーティショニング

1つ以上の列を使ってテーブルをパーティショニングすることで（必要に応じて変換を経由して）、パーティション列でフィルタリングするクエリが全体のファイルをスキップできるようになります。

```rust
// データを読み込む前にパーティションスキームを宣言し、その後通常通りINSERTします。
execute_ducklake_sql(
    &ctx,
    &catalog,
    "ALTER TABLE ducklake.main.sales SET PARTITIONED BY (region, year(sale_date))",
)
.await?;
```

パーティション値ごとに分割された行を別々のParquetファイルに書き出し、読み取り時には一致しないファイルを自動的に除外します。サポートされている変換機能は`identity`（元の値）および`year`/`month`/`day`/`hour`です。現在、除外処理は`identity`および`year`に適用されます（`month`/`day`/`hour`は記録されますが、ファイルのスキップにはまだ使用されません）。**SQLite**では現在、パーティション分けされた書き込みがサポートされており、**すべてのバックエンド**で読み取り時の除外処理が利用可能です。`RESET PARTITIONED BY`を使用するとこの機能は無効になります。詳細は[COMPATIBILITY.md](COMPATIBILITY.md)および[`tests/it/partition_write_tests.rs`](tests/it/partition_write_tests.rs)をご覧ください。

# 厳格な制約事項
1. **構造の維持**：元のMarkdown構造、インデント、見出し階層、表、リンク、URL、バッジ、コードブロック、インラインコードを一切変更しないこと。
2. **選択的翻訳**：ユーザーに表示される可視的な自然言語コンテンツのみを翻訳すること。
3. **変更禁止**：コードラベル、キー名、変数プレースホルダー（{{var}}、${var}、%s、%dなど）、コマンド例、ファイルパス、プロジェクト名、API名、パッケージ名、モデル名、識別子、コード記号の翻訳や変更は**厳禁**である。背景情報に既に対応する訳名が記載されている場合を除く。
4. 用語、スタイル、固有名詞の翻訳は、提供された背景情報と一致させること。

## 複数カタログ機能（PostgreSQL、実験的）

1つのPostgreSQLメタデータストアには、**複数の独立したDuckLakeカタログ**を格納できます。これはマルチテナント環境の構築や、1つのデータベース内に多数の論理的なレイクハウスを保持するのに役立ちます。

> ⚠️ **実験的かつライブラリ専用の機能です。** このマルチカタログ構成は **DuckLake仕様の一部ではなく**、現時点では上位プロジェクトにおいてもサポートされておらず、採用されていません。この方法で作成されたカタログは、このクレートの`MulticatalogProvider`を通じてのみ読み取ることができ、標準的なシングルカタログ型のDuckLakeストアとは互換性がありません。APIやディスク上／カタログ内の構成は変更される可能性があります。現在、PostgreSQLへの書き込みサポートはこの方式に依存しているため、プレビューとして扱ってください。

# 厳格な制約
1. **構造の維持**：元のMarkdownデータ構造、インデント、見出し階層、表、リンク、URL、バッジ、コードブロック、インラインコードを一切変更しないこと。
2. **選択的翻訳**：ユーザーが閲覧する可視的な自然言語内容のみを翻訳すること。
3. **変更禁止**：コードタグ、キー名、変数プレースホルダー（{{var}}、${var}、%s、%dなど）、コマンド例、ファイルパス、プロジェクト名、API名、パッケージ名、モデル名、識別子、コード記号を翻訳したり変更したりすることは**厳禁**である。背景情報に対応する訳名が既に記載されている場合を除く。
4. 用語、スタイル、固有名詞の翻訳は、与えられた背景情報と一致させること。

【翻訳対象部分】
- `MulticatalogManager`を使ってカタログの**作成と管理**を行う（機能`write-postgres`）：
  `initialize_multicatalog_schema`が共有テーブルを初期化し、その後`create_catalog`が実行され、`drop_table_in_catalog`がそれらの内容を管理する。
- 機能`multicatalog-postgres`を備えた`MulticatalogProvider::with_pool(pool, "name")`を使って特定のカタログを**読み取る**。この仕組みは他のメタデータプロバイダと同様に`DuckLakeCatalog`に接続する。

エンドツーエンドの流れ（ブートストラップ → カタログの作成 → 書き込み → 再読み込み）については、
[`examples/multicatalog_write.rs`](examples/multicatalog_write.rs) をご覧ください。

# 厳格な制約事項
1. **構造の維持**：元の Markdown のデータ構造、インデント、見出し階層、表、リンク、URL、バッジ、コードブロック、インラインコードを一切変更しないこと。
2. **選択的翻訳**：ユーザーに表示される可視的な自然言語テキストのみを翻訳すること。
3. **変更禁止**：コードタグ、キー名、変数プレースホルダー（{{var}}、${var}、%s、%d など）、コマンド例、ファイルパス、プロジェクト名、API名、パッケージ名、モデル名、識別子、コード記号を翻訳したり変更したりすることは**厳禁**である。背景情報に対応する訳名が既に記載されている場合を除く。
4. 用語、スタイル、固有名詞の翻訳は、提供された背景情報と一致させること。

## メンテナンス

# 保守管理
`maintenance` APIはRustからレイクハウスの保守作業を処理します。具体的には、古くなったスナップショットの期限切れ処理、廃止されたファイルの削除、孤立したファイルの回収などが行われます。実際の呼び出しポイントはバックエンドによって制御されており（`write-sqlite` / `write-postgres`）、`DROP TABLE`は`MetadataWriter`を通じて利用可能です。詳細は
[`examples/maintenance_demo.rs`](examples/maintenance_demo.rs) および
[`examples/orphan_cleanup_demo.rs`](examples/orphan_cleanup_demo.rs) をご覧ください。

### 圧縮

`DuckLakeTable`上で実行される2つの明示的なトリガー操作により、テーブルの論理的な行は変更されることなく、そのデータファイルがより最適な物理的構造に書き換えられます。

- `merge_adjacent_files(state, MergeOptions)` は、同じスキーマバージョンを持つ複数の小さなファイルをより少ない数の大きなファイルに統合します。複数のオリジンスナップショットにまたがるマージされたファイルは、DuckLake の*部分的データファイル*として書き出され（各行の元の rowid およびオリジンスナップショントが保持されるため）、タイムトラベルやチェンジフィードに影響はありません。
- `rewrite_data_files(state, RewriteOptions)` は、削除された割合が閾値（デフォルトは `0.95`）を超えるファイルを再書き込みし、削除された行を除外します。

これらの操作はいずれも1つのスナップショット内で原子的に実行され、同時に行われる追加操作と共存します。古くなったファイルは削除予定となり、後で`cleanup_old_files`によって再利用されます。詳細は[`examples/compaction_demo.rs`](examples/compaction_demo.rs)をご覧ください。

# 厳格な制約事項
1. **構造の固定**：元のMarkdownデータ構造、インデント、見出しの階層、表、リンク、URL、バッジ、コードブロック、およびインラインコードを一切変更しないこと。
2. **選択的翻訳**：ユーザーに表示される可視的な自然言語コンテンツのみを翻訳すること。
3. **変更禁止**：コードタグ、キー名、変数プレースホルダー（{{var}}、${var}、%s、%dなど）、コマンド例、ファイルパス、プロジェクト名、API名、パッケージ名、モデル名、識別子、およびコード記号の翻訳や変更は**厳禁**である。背景情報に既に対応する訳名が記載されている場合を除く。
4. 用語、文体、固有名詞の翻訳は、提供された背景情報と一致させること。

## 互換性

カタログバックエンド、オブジェクトストレージ、型、機能、および現在の制限事項に関する詳細は、**[COMPATIBILITY.md](COMPATIBILITY.md)** をご覧ください。

事前に知っておくべき主なポイントは以下の通りです：

- 読み取りはDuckDB、SQLite、PostgreSQL、MySQLで動作しますが、**書き込みはSQLite/PostgreSQLのみ**です。  
- オブジェクトストレージとしてはローカルファイルシステムおよびS3互換のストレージ（S3、MinIO）が利用可能です。  
- スナップショットは`DuckLakeCatalog`を通じて選択するか、`ducklake_table_at`を使ってクエリごとに指定できます。DataFusionでは`AS OF`構文はサポートされていません。  
- テーブルのパーティショニングについては、すべてのバックエンドで読み取り時およびファイル削除時に適用され、SQLiteではパーティション別に書き込みが行われます。  
- DuckDBのducklake拡張機能によってインライン化されたデータは**読み取られません**。`COUNT(*)`の値が過小になる問題とその回避方法については、COMPATIBILITY.mdを参照してください。

---

## プロジェクトの状況

このプロジェクトはアルファ版であり、DataFusionおよびDuckLakeと共に進化し続けています。コアな抽象化が洗練されるにつれてAPIが変更される可能性があります。リリース履歴については[CHANGELOG.md](CHANGELOG.md)をご覧ください。フィードバック、問題報告、寄付も歓迎しています。
