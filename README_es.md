# DataFusion-DuckLake

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | **Español**
<!-- hy-mt2-i18n:end -->


[![crates.io](https://img.shields.io/crates/v/datafusion-ducklake.svg)](https://crates.io/crates/datafusion-ducklake)
[![docs.rs](https://img.shields.io/docsrs/datafusion-ducklake)](https://docs.rs/datafusion-ducklake)
[![CI](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml/badge.svg)](https://github.com/hotdata-dev/datafusion-ducklake/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-DataFusion%2BDuckLake-5865F2?logo=discord&logoColor=white)](https://discord.com/channels/885562378132000778/1492192627666321452)

Una extensión de [DataFusion](https://datafusion.apache.org/) para leer y escribir catálogos [DuckLake](https://ducklake.select). DuckLake es un formato integrado de lago de datos y catálogo que almacena los metadatos en una base de datos SQL y los datos en archivos Parquet en el disco o en almacenamiento de objetos.

El objetivo de este proyecto es convertir a DuckLake en un formato lakehouse de primera categoría y nativo de Arrow dentro de DataFusion.

Este proyecto es mantenido por [Hotdata](https://www.hotdata.dev) con el apoyo de la comunidad. Venga a hablar con nosotros en [el Discord de Hotdata](https://discord.gg/cdHczfxxBc).

- 📦 **crates.io:** <https://crates.io/crates/datafusion-ducklake>
- 📖 **Documentación de la API:** <https://docs.rs/datafusion-ducklake>
- 🧩 **Soporte para características y backends:** consulte [COMPATIBILITY.md](COMPATIBILITY.md)
- 💬 **Chat del proyecto:** [DataFusion+DuckLake en Discord](https://discord.com/channels/885562378132000778/1492192627666321452) — discusiones sobre desarrollo y uso
- 🧡 **Conozca al equipo:** [Discord de Hotdata](https://discord.gg/cdHczfxxBc)

# Restricciones estrictas
1. **Bloqueo estructural**: Mantener absolutamente intacta la estructura de datos en Markdown original, el sangrado, los niveles de título, las tablas, los enlaces, las URL, las insignias, los bloques de código y el código inline.
2. **Traducción selectiva**: Solo traducir el contenido de lenguaje natural visible para el usuario.
3. **Prohibición de modificaciones**: Está **estrictamente prohibido** traducir o cambiar etiquetas de código, nombres de clave, placeholders de variables (como {{var}}, ${var}, %s, %d, etc.), ejemplos de comandos, rutas de archivos, nombres de proyectos, nombres de API, nombres de paquetes, nombres de modelos, identificadores y símbolos de código; a menos que la información de contexto ya proporcione su traducción correspondiente.
4. La traducción de términos, estilos y nombres propios debe ser coherente con la información de contexto proporcionada.

## Inicio rápido

Añada el paquete:

```bash
cargo add datafusion-ducklake
```

La compilación por defecto incluye el backend de catálogo DuckDB, empaquetado estáticamente. Otros backends y soporte de escritura se activan de forma opcional mediante flags de características; consulte [COMPATIBILITY.md](COMPATIBILITY.md) para ver la matriz completa.

```toml
# Cargo.toml — leer catálogos de PostgreSQL
# (para la ruta experimental de escritura en múltiples catálogos, utilice features = ["write-postgres"])
[dependencies]
datafusion-ducklake = { version = "0.5", features = ["metadata-postgres"] }
```

Los ejemplos a continuación también utilizan directamente `datafusion`, `object_store` y `url`; añádalos también a su sección `[dependencies]` (este paquete no los vuelve a exportar). El ejemplo de escritura emplea además `sqlx` (con sus características `postgres` y `runtime-tokio`) para abrir el pool de conexiones.

Ejecuta una consulta contra un catálogo PostgreSQL existente con el ejemplo incluido:

```bash
cargo run --example basic_query --features metadata-postgres -- \
  "postgresql://user:password@localhost:5432/database" "SELECT * FROM main.users"
```

(Ejemplo que también acepta cadenas de conexión para DuckDB, SQLite y MySQL, siempre que se utilice la función correspondiente `metadata-*`; consulte [COMPATIBILITY.md](COMPATIBILITY.md).)

# Lectura de un catálogo

## Lectura de un catálogo

Registrar un `DuckLakeCatalog` en un `SessionContext` e interrogarlo con SQL estándar mediante la ruta `catalog.schema.table`:

```rust
ignore
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::*;
use datafusion_ducklake::{DuckLakeCatalog, PostgresMetadataProvider};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;
use url::Url;

// (dentro de una función async)
// Leer metadatos de un catálogo PostgreSQL
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;

// Registrar almacenes de objetos para datos no locales (S3 / MinIO)
let runtime = Arc::new(RuntimeEnv::default());
let s3: Arc<dyn ObjectStore> = Arc::new(
    AmazonS3Builder::new()
       .with_endpoint("http://localhost:9000") // endpoint de MinIO
       .with_bucket_name("ducklake-data")
       .with_access_key_id("minioadmin")
       .with_secret_access_key("minioadmin")
       .with_region("us-west-2") // cualquier región funciona con MinIO
       .with_allow_http(true)    // necesario para endpoints http://
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

## Escritura de un catálogo

## Escritura de un catálogo

PostgreSQL cuenta con dos escritores, ambos respaldados por la función `write-postgres`:

- **`PostgresSingleCatalogMetadataWriter`**: el diseño **estándar y conforme a la especificación** para un único catálogo. Tiene la misma estructura que los escritores de SQLite y MySQL, por lo que puede ser leído (y escrito) por otras implementaciones de DuckLake, incluida la extensión `ducklake` de DuckDB. Funcionan tanto las instrucciones SQL `CREATE TABLE AS SELECT` como `INSERT INTO`. **Se recomienda utilizar este opción.**
- **`PostgresMetadataWriter`**: el diseño **experimental de múltiples catálogos** descrito en [su propia sección](#multi-catalog-postgresql-experimental), diseñado para alojar varios catálogos en una sola base de datos. Es específico de la biblioteca, no forma parte de la especificación de DuckLake, y no admite instrucciones CTAS.

```rust,ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
};
use std::sync::Arc;

// Inicializar las tablas estándar de DuckLake y apuntar el catálogo hacia la raíz de los datos
let writer = PostgresSingleCatalogMetadataWriter::new_with_init(
    "postgresql://user:pass@localhost:5432/db",
).await?;
writer.set_data_path("/abs/path/to/data")?;

// Tanto CTAS como INSERT funcionan en esta ruta
let provider = PostgresMetadataProvider::new("postgresql://user:pass@localhost:5432/db").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("CREATE TABLE ducklake.main.events AS SELECT 1 AS id").await?.collect().await?;
```

La ruta del multicatálogo se ve de esta manera: las tablas se crean mediante la API del escritor (sin CTAS) y luego se añaden con SQL.

```rust,ignore
use datafusion::prelude::*;
use datafusion_ducklake::metadata_writer::MetadataWriter; // set_data_path
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, MulticatalogManager, MulticatalogProvider,
    PostgresMetadataWriter, initialize_multicatalog_schema,
};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;

let pool = PgPoolOptions::new().connect("postgresql://user:pass@localhost:5432/db").await?;

// Se realiza una sola vez: se inicializan las tablas del multicatálogo y luego se crea un catálogo con nombre
initialize_multicatalog_schema(&pool).await?;
let catalog_id = MulticatalogManager::new(pool.clone()).create_catalog("my_catalog").await?;

// Se crea una tabla escribiendo el primer lote mediante el escritor de tablas
let writer = Arc::new(PostgresMetadataWriter::with_pool(pool.clone(), catalog_id).await?);
writer.set_data_path("/abs/path/to/data")?;
let object_store: Arc<dyn object_store::ObjectStore> =
    Arc::new(object_store::local::LocalFileSystem::new());
let table_writer = DuckLakeTableWriter::new(writer.clone(), object_store)?;
table_writer.write_table("public", "events", &[batch]).await?; // `batch` es tu RecordBatch

// Ahora se agrega contenido mediante SQL, leyendo nuevamente del mismo catálogo a través de MulticatalogProvider
let provider = MulticatalogProvider::with_pool(pool.clone(), "my_catalog").await?;
let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), writer)?;
let ctx = SessionContext::new();
ctx.register_catalog("ducklake", Arc::new(catalog));
ctx.sql("INSERT INTO ducklake.public.events VALUES (1, 'a')").await?.collect().await?;
ctx.sql("SELECT count(*) FROM ducklake.public.events").await?.show().await?;
```

La salida del escritor es configurable (compresión Parquet, tamaño de los grupos de filas según la cantidad de filas y el tamaño en bytes). Consulte [`DuckLakeTableWriter`](https://docs.rs/datafusion-ducklake) para conocer las opciones del escritor.

Hoy en día se admite la escritura en un almacén **estándar de un solo catálogo** de DuckLake (el diseño conforme a las especificaciones) para **SQLite** a través de `SqliteMetadataWriter` (funcionalidad `write-sqlite`), donde tanto las instrucciones SQL `CREATE TABLE AS SELECT` como `INSERT INTO` funcionan. Véase [`tests/it/sql_write_tests.rs`](tests/it/sql_write_tests.rs).

# Restricciones estrictas
1. **Bloqueo estructural**: Se debe mantener intacta por completo la estructura de Markdown original, incluyendo el sangrado, los niveles de título, las tablas, los enlaces, las URL, las insignias, los bloques de código y el código incrustado.
2. **Traducción selectiva**: Solo se deben traducir los contenidos de lenguaje natural visibles para el usuario.
3. **Prohibición de modificaciones**: Está **estrictamente prohibido** traducir o cambiar etiquetas de código, nombres de claves, placeholders de variables (como {{var}}, ${var}, %s, %d, etc.), ejemplos de comandos, rutas de archivos, nombres de proyectos, nombres de API, nombres de paquetes, nombres de modelos, identificadores y símbolos de código; a menos que ya exista una traducción correspondiente en la información de contexto.
4. La traducción de términos, estilos y nombres propios debe ser coherente con la información de contexto proporcionada.

## Particionamiento

Particionar una tabla según una o más columnas (opcionalmente mediante una transformación) de modo que las consultas que filtran por una columna de partición omitan archivos completos:

```rust
// Declare el esquema de particiones *antes* de cargar los datos y luego realice el INSERT como de costumbre.
execute_ducklake_sql(
    &ctx,
    &catalog,
    "ALTER TABLE ducklake.main.sales SET PARTITIONED BY (region, year(sale_date))",
)
.await?;
```

Escribe las filas divididas en un archivo Parquet por cada valor de partición; a continuación, lee y elimina automáticamente los archivos que no coinciden. Las transformaciones admitidas son `identity` (el valor original) y `year`/`month`/`day`/`hour`; por el momento, la eliminación se aplica a `identity` y `year` (se registran `month`/`day`/`hour`, pero aún no se utilizan para omitir archivos). Actualmente, las escrituras con particionamiento están disponibles en **SQLite**; la lectura junto con la eliminación funciona en todos los motores de base de datos. `RESET PARTITIONED BY` desactiva esta funcionalidad. Consulte [COMPATIBILITY.md](COMPATIBILITY.md) y [`tests/it/partition_write_tests.rs`](tests/it/partition_write_tests.rs).

## Restricciones estrictas
1. **Bloqueo estructural**: Se debe mantener intacta por completo la estructura de Markdown original, incluyendo el sangrado, los niveles de título, las tablas, los enlaces, las URL, las insignias, los bloques de código y el código inline.
2. **Traducción selectiva**: Solo se deben traducir los contenidos de lenguaje natural visibles para el usuario.
3. **Prohibición de modificaciones**: Está **estrictamente prohibido** traducir o alterar etiquetas de código, nombres de claves, placeholders de variables (como {{var}}, ${var}, %s, %d, etc.), ejemplos de comandos, rutas de archivos, nombres de proyectos, nombres de API, nombres de paquetes, nombres de modelos, identificadores y símbolos de código; a menos que ya exista una traducción correspondiente en la información de contexto.
4. La traducción de términos, estilos y nombres propios debe ser coherente con la información de contexto proporcionada.

## Multi-catálogo (PostgreSQL, experimental)

Un único almacén de metadatos de PostgreSQL puede alojar **múltiples catálogos DuckLake independientes**, lo cual es útil para implementaciones multiinquilino o para mantener muchos lakehouses lógicos en una sola base de datos.

⚠️ **Experimental y específico de la biblioteca.** Este diseño de múltiples catálogos **no forma parte de la especificación de DuckLake** y aún no cuenta con soporte ni está aceptado en el proyecto principal. Los catálogos creados de esta manera solo pueden leerse a través del `MulticatalogProvider` de esta librería; **no son intercambiables** con un almacén estándar de DuckLake con un único catálogo. La API y la estructura en disco o dentro del catálogo pueden cambiar. El soporte para escritura en PostgreSQL depende actualmente de este enfoque, por lo que debe considerarse como una versión preliminar.

- **Crear y gestionar** catálogos con `MulticatalogManager` (funcionalidad `write-postgres`):  
  `initialize_multicatalog_schema` inicializa las tablas compartidas, luego `create_catalog`, y `drop_table_in_catalog` gestiona su contenido.  
- **Leer** un catálogo específico con `MulticatalogProvider::with_pool(pool, "name")`  
  (funcionalidad `multicatalog-postgres`), el cual se integra en un `DuckLakeCatalog` al igual que cualquier otro proveedor de metadatos.

Consulte [`examples/multicatalog_write.rs`](examples/multicatalog_write.rs) para ver una explicación paso a paso completa (arranque inicial → creación de catálogos → escritura → lectura posterior).

## Mantenimiento

## Mantenimiento

La API `maintenance` se encarga del mantenimiento del lakehouse desde Rust: vence la validez de las instantáneas antiguas, elimina los archivos obsoletos y recupera los archivos huérfanos. Los puntos de entrada concretos están controlados por el backend (`write-sqlite` / `write-postgres`). La instrucción `DROP TABLE` está disponible a través de `MetadataWriter`. Consulte
[`examples/maintenance_demo.rs`](examples/maintenance_demo.rs) y
[`examples/orphan_cleanup_demo.rs`](examples/orphan_cleanup_demo.rs).

### Compactación

Dos operaciones explícitas y desencadenables en `DuckLakeTable` reescriben los archivos de datos de una tabla para lograr una estructura física mejor, sin modificar sus filas lógicas:

- `merge_adjacent_files(state, MergeOptions)` combina varios archivos pequeños (de la misma versión de esquema) en unos pocos más grandes. Un archivo fusionado que abarca múltiples instantáneas de origen se escribe como un *archivo de datos parcial* de DuckLake (preservando el rowid original y la instantánea de origen de cada fila), de modo que los viajes en el tiempo y los canales de cambios no se ven afectados.  
- `rewrite_data_files(state, RewriteOptions)` vuelve a escribir un archivo cuya fracción de filas eliminadas supera un umbral (por defecto `0.95`), eliminando dichas filas.

Ambas operaciones se realizan de forma atómica en una sola instantánea y pueden coexistir con inserciones simultáneas; los archivos sobrescritos se programan para su eliminación y serán recuperados posteriormente por `cleanup_old_files`. Consulte [`examples/compaction_demo.rs`](examples/compaction_demo.rs).

## Compatibilidad

Para ver el desglose completo de los backends de catálogo, almacenamientos de objetos, tipos, capacidades y limitaciones actuales, consulte **[COMPATIBILITY.md](COMPATIBILITY.md)**.

Algunos puntos destacados que conviene conocer de antemano:

- Las lecturas funcionan en DuckDB, SQLite, PostgreSQL y MySQL; **las escrituras solo son posibles en SQLite/PostgreSQL**.
- Almacenamientos de objetos: sistema de archivos local y compatibles con S3 (S3, MinIO).

## Compatibilidad

Para conocer en detalle los servidores de catálogo, almacenamientos de objetos, tipos, funcionalidades y limitaciones actuales, consulte **[COMPATIBILITY.md](COMPATIBILITY.md)**.

Algunos puntos destacados que vale la pena conocer de antemano:

- La lectura funciona con DuckDB, SQLite, PostgreSQL y MySQL; **la escritura solo es posible en SQLite/PostgreSQL**.
- Almacenes de objetos: sistema de archivos local y compatibles con S3 (S3, MinIO).
- Las instantáneas se pueden seleccionar mediante `DuckLakeCatalog` o por consulta con `ducklake_table_at`; DataFusion no admite la sintaxis `AS OF`.
- Particionamiento de tablas: lectura más recorte de archivos en todos los backends; escritura particionada en SQLite.
- Los datos incluidos mediante la extensión ducklake de DuckDB **no se leen**; consulte COMPATIBILITY.md para conocer la limitación relacionada con `COUNT(*)` y cómo evitarla.

## Estado del proyecto

Este proyecto se encuentra en versión alfa y evoluciona junto con DataFusion y DuckLake. Las API pueden cambiar a medida que se perfeccionan las abstracciones básicas. Consulte [CHANGELOG.md](CHANGELOG.md) para ver el historial de versiones. Agradecemos los comentarios, los problemas reportados y las contribuciones.

## Estado del proyecto

Este proyecto se encuentra en fase alfa y evoluciona junto con DataFusion y DuckLake. Las API pueden cambiar a medida que se perfeccionan las abstracciones básicas. Consulte [CHANGELOG.md](CHANGELOG.md) para conocer el historial de versiones. Agradecemos cualquier comentario, problema o contribución.
