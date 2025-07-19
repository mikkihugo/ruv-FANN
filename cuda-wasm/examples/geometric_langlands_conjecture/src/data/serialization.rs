//! Efficient serialization and deserialization for mathematical data

use std::path::Path;
use std::io::{BufReader, BufWriter};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use bincode;
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;

use super::{DataError, Result, features::FeatureVector};

/// Serialization format options
#[derive(Debug, Clone, Copy)]
pub enum SerializationFormat {
    /// Binary format using bincode
    Binary,
    /// Compressed binary using gzip
    CompressedBinary,
    /// JSON format (human-readable)
    Json,
    /// MessagePack format
    MessagePack,
}

impl SerializationFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Binary => ".bin",
            Self::CompressedBinary => ".bin.gz",
            Self::Json => ".json",
            Self::MessagePack => ".msgpack",
        }
    }
    
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Self {
        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        match ext {
            "json" => Self::Json,
            "msgpack" => Self::MessagePack,
            "gz" => Self::CompressedBinary,
            _ => Self::Binary,
        }
    }
}

/// Dataset container for batch serialization
#[derive(Serialize, Deserialize)]
pub struct Dataset {
    /// Version for compatibility checking
    pub version: u32,
    
    /// Dataset metadata
    pub metadata: DatasetMetadata,
    
    /// Feature vectors
    pub features: Vec<FeatureVector>,
}

/// Dataset metadata
#[derive(Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Creation timestamp
    pub created_at: u64,
    
    /// Dataset name
    pub name: String,
    
    /// Description
    pub description: String,
    
    /// Number of samples
    pub num_samples: usize,
    
    /// Feature dimensions
    pub geometric_dim: usize,
    pub algebraic_dim: usize,
    
    /// Additional properties
    pub properties: std::collections::HashMap<String, String>,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(name: String, features: Vec<FeatureVector>) -> Self {
        let (geometric_dim, algebraic_dim) = if features.is_empty() {
            (0, 0)
        } else {
            (features[0].geometric.len(), features[0].algebraic.len())
        };
        
        Self {
            version: 1,
            metadata: DatasetMetadata {
                created_at: current_timestamp(),
                name,
                description: String::new(),
                num_samples: features.len(),
                geometric_dim,
                algebraic_dim,
                properties: std::collections::HashMap::new(),
            },
            features,
        }
    }
    
    /// Validate dataset consistency
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(DataError::InvalidFormat(
                format!("Unsupported dataset version: {}", self.version)
            ));
        }
        
        if self.metadata.num_samples != self.features.len() {
            return Err(DataError::InvalidFormat(
                "Metadata sample count doesn't match actual features".to_string()
            ));
        }
        
        // Check dimension consistency
        for (i, feature) in self.features.iter().enumerate() {
            if feature.geometric.len() != self.metadata.geometric_dim {
                return Err(DataError::InvalidFormat(
                    format!("Feature {} has inconsistent geometric dimension", i)
                ));
            }
            if feature.algebraic.len() != self.metadata.algebraic_dim {
                return Err(DataError::InvalidFormat(
                    format!("Feature {} has inconsistent algebraic dimension", i)
                ));
            }
        }
        
        Ok(())
    }
}

/// Load a dataset from file
pub async fn load_dataset(path: &Path) -> Result<Vec<FeatureVector>> {
    let format = SerializationFormat::from_path(path);
    let dataset = load_with_format(path, format).await?;
    dataset.validate()?;
    Ok(dataset.features)
}

/// Save a dataset to file
pub async fn save_dataset(features: &[FeatureVector], path: &Path) -> Result<()> {
    let name = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    
    let dataset = Dataset::new(name, features.to_vec());
    let format = SerializationFormat::from_path(path);
    
    save_with_format(&dataset, path, format).await
}

/// Load dataset with specific format
async fn load_with_format(path: &Path, format: SerializationFormat) -> Result<Dataset> {
    match format {
        SerializationFormat::Binary => load_binary(path).await,
        SerializationFormat::CompressedBinary => load_compressed_binary(path).await,
        SerializationFormat::Json => load_json(path).await,
        SerializationFormat::MessagePack => load_msgpack(path).await,
    }
}

/// Save dataset with specific format
async fn save_with_format(dataset: &Dataset, path: &Path, format: SerializationFormat) 
    -> Result<()> 
{
    match format {
        SerializationFormat::Binary => save_binary(dataset, path).await,
        SerializationFormat::CompressedBinary => save_compressed_binary(dataset, path).await,
        SerializationFormat::Json => save_json(dataset, path).await,
        SerializationFormat::MessagePack => save_msgpack(dataset, path).await,
    }
}

/// Load binary format
async fn load_binary(path: &Path) -> Result<Dataset> {
    let mut file = File::open(path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;
    
    bincode::deserialize(&buffer)
        .map_err(|e| DataError::SerializationError(e))
}

/// Save binary format
async fn save_binary(dataset: &Dataset, path: &Path) -> Result<()> {
    let data = bincode::serialize(dataset)?;
    let mut file = File::create(path).await?;
    file.write_all(&data).await?;
    Ok(())
}

/// Load compressed binary format
async fn load_compressed_binary(path: &Path) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let decoder = GzDecoder::new(reader);
    let reader = BufReader::new(decoder);
    
    bincode::deserialize_from(reader)
        .map_err(|e| DataError::SerializationError(e))
}

/// Save compressed binary format
async fn save_compressed_binary(dataset: &Dataset, path: &Path) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);
        let encoder = GzEncoder::new(writer, Compression::default());
        let writer = BufWriter::new(encoder);
        
        bincode::serialize_into(writer, dataset)
            .map_err(|e| DataError::SerializationError(e))
    }).await?
}

/// Load JSON format
async fn load_json(path: &Path) -> Result<Dataset> {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    
    serde_json::from_str(&contents)
        .map_err(|e| DataError::InvalidFormat(e.to_string()))
}

/// Save JSON format
async fn save_json(dataset: &Dataset, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(dataset)
        .map_err(|e| DataError::InvalidFormat(e.to_string()))?;
    
    let mut file = File::create(path).await?;
    file.write_all(json.as_bytes()).await?;
    Ok(())
}

/// Load MessagePack format
async fn load_msgpack(_path: &Path) -> Result<Dataset> {
    // TODO: Implement MessagePack support
    Err(DataError::InvalidFormat("MessagePack not yet implemented".to_string()))
}

/// Save MessagePack format
async fn save_msgpack(_dataset: &Dataset, _path: &Path) -> Result<()> {
    // TODO: Implement MessagePack support
    Err(DataError::InvalidFormat("MessagePack not yet implemented".to_string()))
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Streaming serializer for large datasets
pub struct StreamingSerializer {
    format: SerializationFormat,
    chunk_size: usize,
}

impl StreamingSerializer {
    pub fn new(format: SerializationFormat, chunk_size: usize) -> Self {
        Self { format, chunk_size }
    }
    
    /// Serialize dataset in chunks
    pub async fn serialize_chunks(
        &self,
        features: Vec<FeatureVector>,
        output_dir: &Path,
        prefix: &str,
    ) -> Result<Vec<String>> {
        let chunks: Vec<_> = features.chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut chunk_files = Vec::new();
        
        for (i, chunk) in chunks.into_iter().enumerate() {
            let filename = format!("{}_chunk_{:04}{}", 
                prefix, i, self.format.extension());
            let path = output_dir.join(&filename);
            
            let dataset = Dataset::new(
                format!("{}_chunk_{}", prefix, i),
                chunk
            );
            
            save_with_format(&dataset, &path, self.format).await?;
            chunk_files.push(filename);
        }
        
        Ok(chunk_files)
    }
    
    /// Load and merge chunks
    pub async fn load_chunks(
        &self,
        chunk_files: Vec<PathBuf>,
    ) -> Result<Vec<FeatureVector>> {
        let mut all_features = Vec::new();
        
        for path in chunk_files {
            let dataset = load_with_format(&path, self.format).await?;
            dataset.validate()?;
            all_features.extend(dataset.features);
        }
        
        Ok(all_features)
    }
}

/// Serialization utilities
pub mod utils {
    use super::*;
    use ndarray::{Array1, Array2};
    
    /// Convert ndarray to Vec for serialization
    pub fn array_to_vec<T: Clone>(arr: &Array1<T>) -> Vec<T> {
        arr.to_vec()
    }
    
    /// Convert Vec back to ndarray
    pub fn vec_to_array<T: Clone>(vec: Vec<T>) -> Array1<T> {
        Array1::from_vec(vec)
    }
    
    /// Serialize sparse matrix efficiently
    pub fn serialize_sparse_matrix(matrix: &Array2<f64>, threshold: f64) 
        -> Vec<(usize, usize, f64)> 
    {
        let mut entries = Vec::new();
        
        for ((i, j), &value) in matrix.indexed_iter() {
            if value.abs() > threshold {
                entries.push((i, j, value));
            }
        }
        
        entries
    }
    
    /// Deserialize sparse matrix
    pub fn deserialize_sparse_matrix(
        entries: Vec<(usize, usize, f64)>,
        shape: (usize, usize),
    ) -> Array2<f64> {
        let mut matrix = Array2::zeros(shape);
        
        for (i, j, value) in entries {
            if i < shape.0 && j < shape.1 {
                matrix[[i, j]] = value;
            }
        }
        
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_dataset_creation() {
        let features = vec![
            FeatureVector::new("test1".to_string(), 10, 5),
            FeatureVector::new("test2".to_string(), 10, 5),
        ];
        
        let dataset = Dataset::new("test_dataset".to_string(), features);
        
        assert_eq!(dataset.metadata.num_samples, 2);
        assert_eq!(dataset.metadata.geometric_dim, 10);
        assert_eq!(dataset.metadata.algebraic_dim, 5);
    }
    
    #[tokio::test]
    async fn test_save_load_binary() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.bin");
        
        let features = vec![
            FeatureVector::new("test1".to_string(), 10, 5),
        ];
        
        save_dataset(&features, &path).await.unwrap();
        let loaded = load_dataset(&path).await.unwrap();
        
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "test1");
    }
    
    #[tokio::test]
    async fn test_format_detection() {
        assert!(matches!(
            SerializationFormat::from_path(Path::new("data.json")),
            SerializationFormat::Json
        ));
        
        assert!(matches!(
            SerializationFormat::from_path(Path::new("data.bin.gz")),
            SerializationFormat::CompressedBinary
        ));
    }
    
    #[tokio::test]
    async fn test_streaming_serializer() {
        let dir = TempDir::new().unwrap();
        
        let features: Vec<_> = (0..10)
            .map(|i| FeatureVector::new(format!("test{}", i), 10, 5))
            .collect();
        
        let serializer = StreamingSerializer::new(SerializationFormat::Binary, 3);
        let chunks = serializer.serialize_chunks(
            features,
            dir.path(),
            "test"
        ).await.unwrap();
        
        assert_eq!(chunks.len(), 4); // 10 features / 3 per chunk = 4 chunks
    }
}