//! In-process S3 server for the object-store tests.
//!
//! By default each [`S3Server::start`] runs its own server: `s3s` speaks the S3
//! REST API (SigV4 auth included) and `s3s-fs` stores objects in a temp
//! directory. Nothing is pulled or started in Docker. MinIO no longer publishes
//! community images: it deleted `minio/*` from Docker Hub on 2026-09-11 and
//! closed `quay.io/minio` on 2026-09-24.
//!
//! Set `TEST_S3_ENDPOINT_URL` to run the same tests against a real
//! S3-compatible server instead (MinIO, RustFS, ...), to check that a behaviour
//! a test relies on is not an artefact of `s3s-fs`. Credentials come from
//! `TEST_S3_USER` / `TEST_S3_PASSWORD`, then the defaults below.

use aws_credential_types::Credentials;
use aws_sdk_s3::config::{Region, SharedCredentialsProvider};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::oneshot;

const DEFAULT_USER: &str = "testadmin";
const DEFAULT_PASSWORD: &str = "testadmin-secret";
pub const REGION: &str = "us-east-1";

/// A running S3 endpoint. Dropping it stops the in-process server and removes
/// its data directory.
pub struct S3Server {
    /// `http://host:port`, with no trailing slash.
    pub endpoint_url: String,
    pub user: String,
    pub password: String,
    _in_process: Option<InProcess>,
}

struct InProcess {
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
    _root: TempDir,
}

impl Drop for InProcess {
    /// Stop the server and wait for its thread, so that no request is still
    /// writing into the root when `_root` removes it after this returns.
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl S3Server {
    /// Start an `s3s-fs` server on a loopback port, or use the server that
    /// `TEST_S3_ENDPOINT_URL` names.
    ///
    /// The in-process server runs on its own thread and runtime rather than the
    /// caller's. A `#[tokio::test]` runtime is single-threaded, and the tests
    /// call DuckDB synchronously on it, so a server spawned there could not
    /// answer the requests DuckDB is blocked on.
    pub fn start() -> Self {
        if let Ok(endpoint_url) = std::env::var("TEST_S3_ENDPOINT_URL") {
            return Self {
                endpoint_url: endpoint_url.trim_end_matches('/').to_string(),
                user: std::env::var("TEST_S3_USER").unwrap_or_else(|_| DEFAULT_USER.into()),
                password: std::env::var("TEST_S3_PASSWORD")
                    .unwrap_or_else(|_| DEFAULT_PASSWORD.into()),
                _in_process: None,
            };
        }

        let root = TempDir::new().expect("failed to create the S3 fixture root");
        let fs = s3s_fs::FileSystem::new(root.path()).expect("failed to open s3s-fs root");
        let service = {
            let mut builder = s3s::service::S3ServiceBuilder::new(fs);
            builder.set_auth(s3s::auth::SimpleAuth::from_single(
                DEFAULT_USER,
                DEFAULT_PASSWORD,
            ));
            builder.build()
        };

        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind the S3 test server");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

        let thread = std::thread::Builder::new()
            .name("s3-test-server".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("failed to build the S3 test server runtime");
                runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    let http = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    );
                    loop {
                        tokio::select! {
                            _ = &mut shutdown_rx => break,
                            accepted = listener.accept() => {
                                let socket = match accepted {
                                    Ok((socket, _)) => socket,
                                    // A persistent error, such as no free file
                                    // descriptors, would spin this loop. Wait
                                    // before the next try instead.
                                    Err(e) => {
                                        eprintln!("S3 test server accept failed: {e}");
                                        tokio::time::sleep(Duration::from_millis(100)).await;
                                        continue;
                                    },
                                };
                                let conn = http
                                    .serve_connection(
                                        hyper_util::rt::TokioIo::new(socket),
                                        service.clone(),
                                    )
                                    .into_owned();
                                tokio::spawn(async move {
                                    let _ = conn.await;
                                });
                            },
                        }
                    }
                });
            })
            .expect("failed to spawn the S3 test server thread");

        Self {
            endpoint_url: format!("http://{addr}"),
            user: DEFAULT_USER.to_string(),
            password: DEFAULT_PASSWORD.to_string(),
            _in_process: Some(InProcess {
                shutdown: Some(shutdown_tx),
                thread: Some(thread),
                _root: root,
            }),
        }
    }

    /// `host:port`, the form DuckDB's `s3_endpoint` setting takes.
    pub fn host_port(&self) -> &str {
        self.endpoint_url
            .trim_start_matches("http://")
            .trim_start_matches("https://")
    }

    /// Whether the endpoint uses TLS.
    pub fn use_ssl(&self) -> bool {
        self.endpoint_url.starts_with("https://")
    }

    /// Create a bucket whose name starts with `prefix`, suffixed so that tests
    /// sharing an external server cannot collide.
    pub async fn create_bucket(&self, prefix: &str) -> anyhow::Result<String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let bucket = format!(
            "{prefix}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let creds = Credentials::new(&self.user, &self.password, None, None, "test");
        let config = aws_sdk_s3::Config::builder()
            .endpoint_url(&self.endpoint_url)
            .region(Region::new(REGION))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            .force_path_style(true)
            .behavior_version_latest()
            .build();
        aws_sdk_s3::Client::from_conf(config)
            .create_bucket()
            .bucket(&bucket)
            .send()
            .await?;
        Ok(bucket)
    }
}
