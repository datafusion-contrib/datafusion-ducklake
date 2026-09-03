//! Encryption support for reading encrypted Parquet files in DuckLake.
//!
//! This module provides integration with Parquet Modular Encryption (PME) using
//! encryption keys stored in the DuckLake catalog.
//!
//! # Supported Encryption
//!
//! This module supports **PME-compliant** encrypted Parquet files created by:
//! - Apache Arrow (PyArrow, arrow-rs)
//! - Apache Spark
//! - Any tool following the Parquet Modular Encryption specification
//!
//! # NOT Supported
//!
//! **DuckDB-encrypted Parquet files are NOT supported.** DuckDB's encryption
//! implementation is non-standard and missing required PME fields (`aad_file_unique`).
//! Attempting to read DuckDB-encrypted files will result in an error:
//! "AAD unique file identifier is not set"
//!
//! For more information, see: <https://duckdb.org/docs/stable/data/parquet/encryption>

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Error message for DuckDB-encrypted files (non-PME compliant)
pub const DUCKDB_ENCRYPTION_ERROR: &str = "Encrypted Parquet file detected but decryption failed. \
     This may be a DuckDB-encrypted file which uses non-standard encryption \
     (missing 'aad_file_unique' field required by PME specification). \
     DataFusion can only decrypt PME-compliant files created by PyArrow, Spark, or parquet-rs. \
     See: https://duckdb.org/docs/stable/data/parquet/encryption";

/// Known error patterns from parquet-rs that indicate DuckDB encryption
const DUCKDB_ERROR_PATTERNS: &[&str] =
    &["AAD unique file identifier is not set", "aad_file_unique"];

/// Check if an error message indicates a DuckDB encryption incompatibility
pub fn is_duckdb_encryption_error(error_msg: &str) -> bool {
    DUCKDB_ERROR_PATTERNS
        .iter()
        .any(|pattern| error_msg.contains(pattern))
}

/// Wrap a parquet/datafusion error with a more helpful message if it's a DuckDB encryption error
#[cfg(feature = "encryption")]
pub fn wrap_encryption_error(
    error: datafusion::error::DataFusionError,
) -> datafusion::error::DataFusionError {
    let error_str = error.to_string();
    if is_duckdb_encryption_error(&error_str) {
        datafusion::error::DataFusionError::Execution(format!(
            "{}. Original error: {}",
            DUCKDB_ENCRYPTION_ERROR, error_str
        ))
    } else {
        error
    }
}

#[cfg(feature = "encryption")]
use arrow::datatypes::SchemaRef;
#[cfg(feature = "encryption")]
use async_trait::async_trait;
#[cfg(feature = "encryption")]
use datafusion::common::config::EncryptionFactoryOptions;
#[cfg(feature = "encryption")]
use datafusion::error::Result;
#[cfg(feature = "encryption")]
use datafusion::execution::parquet_encryption::EncryptionFactory;
#[cfg(feature = "encryption")]
use object_store::path::Path;
#[cfg(feature = "encryption")]
use parquet::encryption::decrypt::FileDecryptionProperties;
#[cfg(feature = "encryption")]
use parquet::encryption::encrypt::FileEncryptionProperties;

/// DuckLake encryption factory that provides decryption keys from the catalog.
///
/// This factory maintains a mapping of file paths to their encryption keys,
/// populated from the DuckLake catalog metadata.
///
/// The map sits behind a lock so that a caller listing files in pages can hand
/// out one factory over a key set it goes on adding to ([`Self::shared`]).
/// Readers that already cloned the factory resolve keys installed after them,
/// and there is no second copy of the map to fall out of step with the one
/// being added to.
#[derive(Clone)]
pub struct DuckLakeEncryptionFactory {
    /// Map of file paths to their encryption keys (base64 or hex encoded)
    file_keys: Arc<RwLock<HashMap<String, String>>>,
}

// Custom Debug implementation to avoid exposing encryption keys in logs
impl std::fmt::Debug for DuckLakeEncryptionFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("DuckLakeEncryptionFactory");
        match self.file_keys.read() {
            Ok(file_keys) => out
                .field("file_count", &file_keys.len())
                .field("files", &file_keys.keys().collect::<Vec<_>>()),
            // Unreadable only if a writer panicked holding the lock. Formatting
            // must not panic in turn, or the panic that poisoned it cannot be
            // reported.
            Err(_) => out.field("file_count", &"poisoned"),
        }
        .finish()
    }
}

impl DuckLakeEncryptionFactory {
    /// Create a new encryption factory with the given file-to-key mapping.
    ///
    /// # Arguments
    /// * `file_keys` - Map of file paths to encryption keys
    pub fn new(file_keys: HashMap<String, String>) -> Self {
        Self::shared(Arc::new(RwLock::new(file_keys)))
    }

    /// Create a factory over a key map its owner goes on adding to.
    ///
    /// The factory looks keys up through that same map, so a path installed
    /// after this call resolves through a factory handed out before it. That is
    /// what lets a caller listing files in bounded pages keep one factory for
    /// the whole listing instead of rebuilding it over the accumulated keys
    /// every page.
    ///
    /// # Arguments
    /// * `file_keys` - Shared map of file paths to encryption keys
    pub fn shared(file_keys: Arc<RwLock<HashMap<String, String>>>) -> Self {
        Self {
            file_keys,
        }
    }

    /// Create an empty encryption factory (for tables with no encrypted files).
    pub fn empty() -> Self {
        Self::new(HashMap::new())
    }

    /// Check if any files have encryption keys.
    pub fn has_encrypted_files(&self) -> bool {
        !self.file_keys.read().unwrap().is_empty()
    }

    /// Decode an encryption key from its stored format.
    ///
    /// DuckLake stores keys as strings - this function handles decoding.
    /// Supports: base64, hex, or raw 16/24/32-byte keys.
    ///
    /// Decoding priority:
    /// 1. Base64 (if decodes to valid AES length)
    /// 2. Hex (if decodes to valid AES length)
    /// 3. Raw bytes (if exactly 16, 24, or 32 chars)
    #[cfg(feature = "encryption")]
    pub fn decode_key(key: &str) -> Result<Vec<u8>> {
        use base64::Engine;
        use datafusion::error::DataFusionError;

        // Try base64 first (most common for DuckLake)
        // Falls through to hex if base64 decode fails or produces invalid AES length
        if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(key)
            && Self::is_valid_aes_length(decoded.len())
        {
            return Ok(decoded);
        }

        // Try hex encoding as fallback
        // Falls through to raw bytes if hex decode fails or produces invalid AES length
        if let Ok(decoded) = hex::decode(key)
            && Self::is_valid_aes_length(decoded.len())
        {
            return Ok(decoded);
        }

        // If it's exactly 16, 24, or 32 bytes, treat as raw key
        let bytes = key.as_bytes();
        if Self::is_valid_aes_length(bytes.len()) {
            return Ok(bytes.to_vec());
        }

        // Provide specific error message (without exposing the key value)
        Err(DataFusionError::Execution(format!(
            "Invalid encryption key format. Expected: base64-encoded (recommended), \
             hex-encoded, or raw 16/24/32-byte string. Provided key length: {} chars. \
             Hint: Use base64 encoding, e.g., 'MDEyMzQ1Njc4OWFiY2RlZg==' for a 16-byte key.",
            key.len()
        )))
    }

    /// Check if a key length is valid for AES (128, 192, or 256 bits)
    #[cfg(feature = "encryption")]
    fn is_valid_aes_length(len: usize) -> bool {
        matches!(len, 16 | 24 | 32)
    }

    /// Validate that decoded key is valid AES key length (128 or 256 bits)
    /// Used by tests to verify error message format
    #[cfg(all(feature = "encryption", test))]
    fn validate_key_length(key: Vec<u8>) -> Result<Vec<u8>> {
        use datafusion::error::DataFusionError;

        match key.len() {
            16 => Ok(key), // AES-128
            32 => Ok(key), // AES-256
            24 => Ok(key), // AES-192 (less common but valid)
            len => Err(DataFusionError::Execution(format!(
                "Invalid encryption key length: {} bytes. AES requires 16 bytes (128-bit), \
                 24 bytes (192-bit), or 32 bytes (256-bit)",
                len
            ))),
        }
    }
}

#[cfg(feature = "encryption")]
#[async_trait]
impl EncryptionFactory for DuckLakeEncryptionFactory {
    async fn get_file_encryption_properties(
        &self,
        _config: &EncryptionFactoryOptions,
        _schema: &SchemaRef,
        _file_path: &Path,
    ) -> Result<Option<Arc<FileEncryptionProperties>>> {
        // DuckLake is read-only for now, so we don't support writing encrypted files
        Ok(None)
    }

    async fn get_file_decryption_properties(
        &self,
        _config: &EncryptionFactoryOptions,
        file_path: &Path,
    ) -> Result<Option<Arc<FileDecryptionProperties>>> {
        let path_str = file_path.to_string();

        // Try to find the key for this file path
        // The path might be stored with or without leading slash, so try multiple formats:
        // 1. Exact match
        // 2. With leading slash added
        // 3. With leading slash removed
        // 4. Normalized comparison (both without leading slash)
        // The key is copied out under the read lock rather than held across the
        // decode below, so an installing writer is never blocked on a decode.
        let key = {
            let file_keys = self.file_keys.read().unwrap();
            file_keys
                .get(&path_str)
                .or_else(|| file_keys.get(&format!("/{}", path_str)))
                .or_else(|| file_keys.get(path_str.trim_start_matches('/')))
                .or_else(|| {
                    // Try matching by normalized paths - useful when absolute paths differ
                    // e.g., stored as "/Users/x/data/file.parquet" but received as "Users/x/data/file.parquet"
                    file_keys.iter().find_map(|(stored_path, key)| {
                        let stored_normalized = stored_path.trim_start_matches('/');
                        let path_normalized = path_str.trim_start_matches('/');
                        if stored_normalized == path_normalized {
                            Some(key)
                        } else {
                            None
                        }
                    })
                })
                .cloned()
        };

        match key {
            Some(encoded_key) => {
                let key_bytes = Self::decode_key(&encoded_key)?;

                // Create decryption properties with the footer key
                // DuckLake uses uniform encryption (same key for footer and all columns)
                let props = FileDecryptionProperties::builder(key_bytes)
                    .build()
                    .map_err(|e| {
                        datafusion::error::DataFusionError::Execution(format!(
                            "Failed to create decryption properties: {}",
                            e
                        ))
                    })?;

                Ok(Some(props))
            },
            None => {
                // No encryption key for this file - it's not encrypted
                Ok(None)
            },
        }
    }
}

/// Builder for creating a DuckLakeEncryptionFactory from file metadata.
#[derive(Debug, Default)]
pub struct EncryptionFactoryBuilder {
    file_keys: HashMap<String, String>,
}

impl EncryptionFactoryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an encryption key for a file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file
    /// * `encryption_key` - The encryption key (if Some)
    pub fn add_file(&mut self, file_path: &str, encryption_key: Option<&str>) -> &mut Self {
        if let Some(key) = encryption_key.filter(|k| !k.is_empty()) {
            self.file_keys
                .insert(file_path.to_string(), key.to_string());
        }
        self
    }

    /// Build the encryption factory.
    pub fn build(self) -> DuckLakeEncryptionFactory {
        DuckLakeEncryptionFactory::new(self.file_keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Factory Construction Tests ====================

    #[test]
    fn test_empty_factory() {
        let factory = DuckLakeEncryptionFactory::empty();
        assert!(!factory.has_encrypted_files());
        assert_eq!(factory.file_keys.read().unwrap().len(), 0);
    }

    #[test]
    fn test_factory_with_keys() {
        let mut keys = HashMap::new();
        keys.insert("file1.parquet".to_string(), "key1".to_string());
        keys.insert("file2.parquet".to_string(), "key2".to_string());

        let factory = DuckLakeEncryptionFactory::new(keys);
        assert!(factory.has_encrypted_files());
        assert_eq!(factory.file_keys.read().unwrap().len(), 2);
    }

    #[test]
    fn test_builder_basic() {
        let mut builder = EncryptionFactoryBuilder::new();
        builder.add_file("file1.parquet", Some("key1"));
        builder.add_file("file2.parquet", None);
        builder.add_file("file3.parquet", Some(""));
        let factory = builder.build();

        assert!(factory.has_encrypted_files());
        assert_eq!(factory.file_keys.read().unwrap().len(), 1);
    }

    #[test]
    fn test_builder_multiple_files() {
        let mut builder = EncryptionFactoryBuilder::new();
        builder.add_file("/path/to/file1.parquet", Some("a]key1"));
        builder.add_file("/path/to/file2.parquet", Some("key2"));
        builder.add_file("/path/to/file3.parquet", Some("key3"));
        let factory = builder.build();

        assert_eq!(factory.file_keys.read().unwrap().len(), 3);
        assert!(
            factory
                .file_keys
                .read()
                .unwrap()
                .contains_key("/path/to/file1.parquet")
        );
        assert!(
            factory
                .file_keys
                .read()
                .unwrap()
                .contains_key("/path/to/file2.parquet")
        );
        assert!(
            factory
                .file_keys
                .read()
                .unwrap()
                .contains_key("/path/to/file3.parquet")
        );
    }

    #[test]
    fn test_builder_overwrites_duplicate_paths() {
        let mut builder = EncryptionFactoryBuilder::new();
        builder.add_file("file.parquet", Some("key1"));
        builder.add_file("file.parquet", Some("key2"));
        let factory = builder.build();

        assert_eq!(factory.file_keys.read().unwrap().len(), 1);
        assert_eq!(
            factory.file_keys.read().unwrap().get("file.parquet"),
            Some(&"key2".to_string())
        );
    }

    // ==================== DuckDB Error Detection Tests ====================

    #[test]
    fn test_is_duckdb_encryption_error_aad_unique() {
        assert!(is_duckdb_encryption_error(
            "AAD unique file identifier is not set"
        ));
        assert!(is_duckdb_encryption_error(
            "Parquet error: AAD unique file identifier is not set"
        ));
        assert!(is_duckdb_encryption_error(
            "Some prefix: AAD unique file identifier is not set - some suffix"
        ));
    }

    #[test]
    fn test_is_duckdb_encryption_error_aad_file_unique() {
        assert!(is_duckdb_encryption_error(
            "aad_file_unique field is missing"
        ));
        assert!(is_duckdb_encryption_error(
            "Error: aad_file_unique not found"
        ));
    }

    #[test]
    fn test_is_not_duckdb_encryption_error() {
        assert!(!is_duckdb_encryption_error("Invalid key format"));
        assert!(!is_duckdb_encryption_error("File not found"));
        assert!(!is_duckdb_encryption_error("Decryption failed"));
        assert!(!is_duckdb_encryption_error(""));
    }

    // ==================== Debug Implementation Tests ====================

    #[test]
    fn test_debug_does_not_expose_keys() {
        let mut keys = HashMap::new();
        keys.insert("file1.parquet".to_string(), "SECRET_KEY_12345".to_string());
        keys.insert("file2.parquet".to_string(), "ANOTHER_SECRET".to_string());

        let factory = DuckLakeEncryptionFactory::new(keys);
        let debug_output = format!("{:?}", factory);

        // Debug should show file paths but NOT key values
        assert!(debug_output.contains("file1.parquet"));
        assert!(debug_output.contains("file2.parquet"));
        assert!(debug_output.contains("file_count"));
        assert!(!debug_output.contains("SECRET_KEY_12345"));
        assert!(!debug_output.contains("ANOTHER_SECRET"));
    }

    #[test]
    fn test_debug_shows_file_count() {
        let mut keys = HashMap::new();
        keys.insert("a.parquet".to_string(), "key".to_string());
        keys.insert("b.parquet".to_string(), "key".to_string());
        keys.insert("c.parquet".to_string(), "key".to_string());

        let factory = DuckLakeEncryptionFactory::new(keys);
        let debug_output = format!("{:?}", factory);

        assert!(debug_output.contains("file_count: 3"));
    }

    // ==================== Key Validation Tests (encryption feature) ====================

    #[cfg(feature = "encryption")]
    mod key_validation {
        use super::*;

        #[test]
        fn test_decode_key_base64_valid_16_bytes() {
            // "0123456789abcdef" base64-encoded = 16 bytes
            let key = "MDEyMzQ1Njc4OWFiY2RlZg==";
            let result = DuckLakeEncryptionFactory::decode_key(key);
            assert!(result.is_ok());
            let decoded = result.unwrap();
            assert_eq!(decoded.len(), 16);
            assert_eq!(&decoded, b"0123456789abcdef");
        }

        #[test]
        fn test_decode_key_base64_valid_32_bytes() {
            // 32 random bytes base64-encoded
            use base64::Engine;
            let original_key = vec![0u8; 32]; // 32 zero bytes for AES-256
            let encoded = base64::engine::general_purpose::STANDARD.encode(&original_key);
            let result = DuckLakeEncryptionFactory::decode_key(&encoded);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), 32);
        }

        #[test]
        fn test_decode_key_hex_valid() {
            // 48 hex chars = 24 bytes (AES-192)
            // Base64 decode of 48 alphanumeric chars = 36 bytes (invalid AES)
            // So base64 decode succeeds but validation fails, falls through to hex
            let key = "000102030405060708090a0b0c0d0e0f1011121314151617";
            let result = DuckLakeEncryptionFactory::decode_key(key);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), 24); // AES-192
        }

        #[test]
        fn test_decode_key_raw_16_bytes() {
            // Use chars that are NOT valid base64 to ensure raw fallback is tested
            // Base64 alphabet: A-Z, a-z, 0-9, +, /, =
            // Using special chars that will fail base64 decode
            let key = "!@#$%^&*()_+{}[]"; // Exactly 16 chars, not valid base64 or hex
            let result = DuckLakeEncryptionFactory::decode_key(key);
            assert!(result.is_ok());
            let decoded = result.unwrap();
            assert_eq!(decoded.len(), 16);
            assert_eq!(&decoded, key.as_bytes());
        }

        #[test]
        fn test_decode_key_raw_32_bytes() {
            // 32 chars with special characters that aren't base64 or hex
            let key = "!@#$%^&*()_+{}[]!@#$%^&*()_+{}[]";
            let result = DuckLakeEncryptionFactory::decode_key(key);
            assert!(result.is_ok());
            let decoded = result.unwrap();
            assert_eq!(decoded.len(), 32);
            assert_eq!(&decoded, key.as_bytes());
        }

        #[test]
        fn test_decode_key_invalid_format_and_length() {
            // Not valid base64, not valid hex, and not 16 or 32 chars
            let key = "!@#short"; // 8 chars
            let result = DuckLakeEncryptionFactory::decode_key(key);
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("Invalid encryption key format"));
        }

        #[test]
        fn test_decode_key_base64_invalid_aes_length() {
            // Valid base64 that decodes to 50 bytes (not valid AES length)
            // The encoded string will be ~68 chars, which also fails raw check
            use base64::Engine;
            let fifty_bytes = vec![1u8; 50];
            let encoded = base64::engine::general_purpose::STANDARD.encode(&fifty_bytes);
            // encoded is 68 chars, base64 decodes to 50 bytes (invalid)
            // hex decode would be 34 bytes (invalid), raw is 68 chars (invalid)
            let result = DuckLakeEncryptionFactory::decode_key(&encoded);
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("Invalid encryption key format"));
        }

        #[test]
        fn test_validate_key_length_valid() {
            assert!(DuckLakeEncryptionFactory::validate_key_length(vec![0u8; 16]).is_ok());
            assert!(DuckLakeEncryptionFactory::validate_key_length(vec![0u8; 24]).is_ok());
            assert!(DuckLakeEncryptionFactory::validate_key_length(vec![0u8; 32]).is_ok());
        }

        #[test]
        fn test_validate_key_length_invalid() {
            let result = DuckLakeEncryptionFactory::validate_key_length(vec![0u8; 10]);
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("Invalid encryption key length: 10 bytes"));
        }
    }

    // ==================== Error Message Tests ====================

    #[test]
    fn test_duckdb_error_message_content() {
        assert!(DUCKDB_ENCRYPTION_ERROR.contains("DuckDB"));
        assert!(DUCKDB_ENCRYPTION_ERROR.contains("PME"));
        assert!(DUCKDB_ENCRYPTION_ERROR.contains("aad_file_unique"));
        assert!(DUCKDB_ENCRYPTION_ERROR.contains("PyArrow"));
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn test_wrap_encryption_error_duckdb() {
        let original = datafusion::error::DataFusionError::Execution(
            "AAD unique file identifier is not set".to_string(),
        );
        let wrapped = wrap_encryption_error(original);
        let msg = wrapped.to_string();

        assert!(msg.contains("DuckDB"));
        assert!(msg.contains("PME"));
        assert!(msg.contains("Original error"));
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn test_wrap_encryption_error_passthrough() {
        let original =
            datafusion::error::DataFusionError::Execution("Some other error".to_string());
        let wrapped = wrap_encryption_error(original);
        let msg = wrapped.to_string();

        assert!(!msg.contains("DuckDB"));
        assert!(msg.contains("Some other error"));
    }
}
