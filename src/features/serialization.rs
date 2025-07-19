//! Efficient serialization and deserialization for feature vectors
//!
//! This module provides high-performance serialization with compression,
//! format flexibility, and data integrity checks.

use crate::features::{FeatureVector, FeatureResult, FeatureError};
use std::io::{Read, Write};

/// Serialization formats supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SerializationFormat {
    /// Binary format with optional compression
    Binary,
    /// JSON format (human-readable)
    Json,
    /// MessagePack (compact binary)
    MessagePack,
    /// Custom optimized format
    Custom,
}

impl Default for SerializationFormat {
    fn default() -> Self {
        SerializationFormat::Binary
    }
}

/// Compression algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zlib compression
    Zlib,
    /// LZ4 compression (fast)
    Lz4,
    /// ZSTD compression (balanced)
    Zstd,
}

impl Default for CompressionType {
    fn default() -> Self {
        CompressionType::Gzip
    }
}

/// Configuration for serialization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SerializationConfig {
    /// Serialization format
    pub format: SerializationFormat,
    /// Compression type
    pub compression: CompressionType,
    /// Compression level (0-9, 6 is default)
    pub compression_level: u32,
    /// Include metadata in serialization
    pub include_metadata: bool,
    /// Include feature labels
    pub include_labels: bool,
    /// Validate data integrity on deserialization
    pub validate_integrity: bool,
    /// Pretty print for human-readable formats
    pub pretty_print: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Binary,
            compression: CompressionType::Gzip,
            compression_level: 6,
            include_metadata: true,
            include_labels: true,
            validate_integrity: true,
            pretty_print: false,
        }
    }
}

/// Feature vector serializer
#[derive(Debug, Clone)]
pub struct FeatureSerializer {
    config: SerializationConfig,
}

impl FeatureSerializer {
    /// Create a new serializer with default configuration
    pub fn new() -> Self {
        Self {
            config: SerializationConfig::default(),
        }
    }
    
    /// Create a serializer with custom configuration
    pub fn with_config(config: SerializationConfig) -> Self {
        Self { config }
    }
    
    /// Serialize a feature vector to bytes
    pub fn serialize(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        let serialized = match self.config.format {
            SerializationFormat::Binary => self.serialize_binary(features)?,
            SerializationFormat::Json => self.serialize_json(features)?,
            SerializationFormat::MessagePack => self.serialize_messagepack(features)?,
            SerializationFormat::Custom => self.serialize_custom(features)?,
        };
        
        // Apply compression if specified
        if self.config.compression != CompressionType::None {
            self.compress(&serialized)
        } else {
            Ok(serialized)
        }
    }
    
    /// Deserialize bytes to a feature vector
    pub fn deserialize(&self, data: &[u8]) -> FeatureResult<FeatureVector> {
        // Decompress if necessary
        let decompressed = if self.config.compression != CompressionType::None {
            self.decompress(data)?
        } else {
            data.to_vec()
        };
        
        let features = match self.config.format {
            SerializationFormat::Binary => self.deserialize_binary(&decompressed)?,
            SerializationFormat::Json => self.deserialize_json(&decompressed)?,
            SerializationFormat::MessagePack => self.deserialize_messagepack(&decompressed)?,
            SerializationFormat::Custom => self.deserialize_custom(&decompressed)?,
        };
        
        // Validate integrity if enabled
        if self.config.validate_integrity {
            self.validate_deserialized(&features)?;
        }
        
        Ok(features)
    }
    
    /// Serialize to a writer
    pub fn serialize_to_writer<W: Write>(
        &self,
        features: &FeatureVector,
        writer: &mut W,
    ) -> FeatureResult<()> {
        let serialized = self.serialize(features)?;
        writer.write_all(&serialized)
            .map_err(|e| FeatureError::SerializationFailed(e.to_string()))?;
        Ok(())
    }
    
    /// Deserialize from a reader
    pub fn deserialize_from_reader<R: Read>(
        &self,
        reader: &mut R,
    ) -> FeatureResult<FeatureVector> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        self.deserialize(&data)
    }
    
    /// Batch serialize multiple feature vectors
    pub fn serialize_batch(&self, features: &[FeatureVector]) -> FeatureResult<Vec<u8>> {
        let mut serialized_batch = Vec::new();
        
        // Write header
        serialized_batch.extend_from_slice(&(features.len() as u64).to_le_bytes());
        
        // Serialize each feature vector
        for fv in features {
            let serialized = self.serialize(fv)?;
            serialized_batch.extend_from_slice(&(serialized.len() as u64).to_le_bytes());
            serialized_batch.extend_from_slice(&serialized);
        }
        
        Ok(serialized_batch)
    }
    
    /// Batch deserialize multiple feature vectors
    pub fn deserialize_batch(&self, data: &[u8]) -> FeatureResult<Vec<FeatureVector>> {
        if data.len() < 8 {
            return Err(FeatureError::DeserializationFailed(
                "Insufficient data for batch header".to_string()
            ));
        }
        
        let mut offset = 0;
        
        // Read batch size
        let batch_size = u64::from_le_bytes(
            data[offset..offset + 8].try_into()
                .map_err(|_| FeatureError::DeserializationFailed(
                    "Invalid batch size encoding".to_string()
                ))?
        ) as usize;
        offset += 8;
        
        let mut features = Vec::with_capacity(batch_size);
        
        // Deserialize each feature vector
        for _ in 0..batch_size {
            if offset + 8 > data.len() {
                return Err(FeatureError::DeserializationFailed(
                    "Insufficient data for feature vector size".to_string()
                ));
            }
            
            // Read size of this feature vector
            let fv_size = u64::from_le_bytes(
                data[offset..offset + 8].try_into()
                    .map_err(|_| FeatureError::DeserializationFailed(
                        "Invalid feature vector size encoding".to_string()
                    ))?
            ) as usize;
            offset += 8;
            
            if offset + fv_size > data.len() {
                return Err(FeatureError::DeserializationFailed(
                    "Insufficient data for feature vector".to_string()
                ));
            }
            
            // Deserialize feature vector
            let fv_data = &data[offset..offset + fv_size];
            let fv = self.deserialize(fv_data)?;
            features.push(fv);
            offset += fv_size;
        }
        
        Ok(features)
    }
    
    // Private serialization methods
    
    #[cfg(feature = "serde")]
    fn serialize_binary(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        bincode::serialize(features)
            .map_err(|e| FeatureError::SerializationFailed(e.to_string()))
    }
    
    #[cfg(not(feature = "serde"))]
    fn serialize_binary(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        // Custom binary serialization without serde
        let mut data = Vec::new();
        
        // Write dimension
        data.extend_from_slice(&(features.dimension as u64).to_le_bytes());
        
        // Write values
        for &value in &features.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        // Write metadata if enabled
        if self.config.include_metadata {
            let metadata_json = format!(
                "{{\"object_type\":\"{}\",\"encoding_strategy\":\"{}\",\"timestamp\":{}}}",
                features.metadata.object_type,
                features.metadata.encoding_strategy,
                features.metadata.timestamp.unwrap_or(0)
            );
            let metadata_bytes = metadata_json.into_bytes();
            data.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(&metadata_bytes);
        } else {
            data.extend_from_slice(&0u64.to_le_bytes());
        }
        
        // Write labels if enabled
        if self.config.include_labels && features.labels.is_some() {
            let labels = features.labels.as_ref().unwrap();
            let labels_json = format!("[{}]", 
                labels.iter()
                    .map(|l| format!("\"{}\"", l))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let labels_bytes = labels_json.into_bytes();
            data.extend_from_slice(&(labels_bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(&labels_bytes);
        } else {
            data.extend_from_slice(&0u64.to_le_bytes());
        }
        
        Ok(data)
    }
    
    #[cfg(feature = "serde")]
    fn serialize_json(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        let json_str = if self.config.pretty_print {
            serde_json::to_string_pretty(features)
        } else {
            serde_json::to_string(features)
        }.map_err(|e| FeatureError::SerializationFailed(e.to_string()))?;
        
        Ok(json_str.into_bytes())
    }
    
    #[cfg(not(feature = "serde"))]
    fn serialize_json(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        // Simple JSON serialization without serde
        let values_str = features.values.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        
        let json = format!(
            "{{\"values\":[{}],\"dimension\":{},\"metadata\":{{\"object_type\":\"{}\",\"encoding_strategy\":\"{}\"}}}}",
            values_str,
            features.dimension,
            features.metadata.object_type,
            features.metadata.encoding_strategy
        );
        
        Ok(json.into_bytes())
    }
    
    fn serialize_messagepack(&self, _features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        // MessagePack serialization would require the rmp-serde crate
        Err(FeatureError::SerializationFailed(
            "MessagePack serialization not implemented".to_string()
        ))
    }
    
    fn serialize_custom(&self, features: &FeatureVector) -> FeatureResult<Vec<u8>> {
        // Custom optimized format
        let mut data = Vec::new();
        
        // Magic header
        data.extend_from_slice(b"RUVF"); // RUV Features
        
        // Version
        data.push(1);
        
        // Flags
        let mut flags = 0u8;
        if self.config.include_metadata { flags |= 0x01; }
        if self.config.include_labels && features.labels.is_some() { flags |= 0x02; }
        data.push(flags);
        
        // Dimension (variable length encoding)
        self.write_varint(&mut data, features.dimension as u64);
        
        // Values (optimized based on data characteristics)
        self.write_optimized_values(&mut data, &features.values)?;
        
        // Metadata if enabled
        if self.config.include_metadata {
            self.write_string(&mut data, &features.metadata.object_type);
            self.write_string(&mut data, &features.metadata.encoding_strategy);
            data.extend_from_slice(&features.metadata.timestamp.unwrap_or(0).to_le_bytes());
        }
        
        // Labels if enabled
        if flags & 0x02 != 0 {
            let labels = features.labels.as_ref().unwrap();
            self.write_varint(&mut data, labels.len() as u64);
            for label in labels {
                self.write_string(&mut data, label);
            }
        }
        
        Ok(data)
    }
    
    // Private deserialization methods
    
    #[cfg(feature = "serde")]
    fn deserialize_binary(&self, data: &[u8]) -> FeatureResult<FeatureVector> {
        bincode::deserialize(data)
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))
    }
    
    #[cfg(not(feature = "serde"))]
    fn deserialize_binary(&self, data: &[u8]) -> FeatureResult<Vec<u8>> {
        let mut offset = 0;
        
        // Read dimension
        if data.len() < 8 {
            return Err(FeatureError::DeserializationFailed(
                "Insufficient data for dimension".to_string()
            ));
        }
        let dimension = u64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap()
        ) as usize;
        offset += 8;
        
        // Read values
        let values_size = dimension * 8;
        if data.len() < offset + values_size {
            return Err(FeatureError::DeserializationFailed(
                "Insufficient data for values".to_string()
            ));
        }
        
        let mut values = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let value_bytes = &data[offset + i * 8..offset + (i + 1) * 8];
            let value = f64::from_le_bytes(value_bytes.try_into().unwrap());
            values.push(value);
        }
        offset += values_size;
        
        // Read metadata length
        if data.len() < offset + 8 {
            return Err(FeatureError::DeserializationFailed(
                "Insufficient data for metadata length".to_string()
            ));
        }
        let metadata_len = u64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap()
        ) as usize;
        offset += 8;
        
        // Read metadata if present
        let (object_type, encoding_strategy) = if metadata_len > 0 {
            if data.len() < offset + metadata_len {
                return Err(FeatureError::DeserializationFailed(
                    "Insufficient data for metadata".to_string()
                ));
            }
            
            let metadata_str = String::from_utf8(
                data[offset..offset + metadata_len].to_vec()
            ).map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
            
            // Simple JSON parsing (would be better with a proper parser)
            let object_type = metadata_str
                .split('"')
                .nth(3)
                .unwrap_or("unknown")
                .to_string();
            let encoding_strategy = metadata_str
                .split('"')
                .nth(7)
                .unwrap_or("unknown")
                .to_string();
            
            offset += metadata_len;
            (object_type, encoding_strategy)
        } else {
            ("unknown".to_string(), "unknown".to_string())
        };
        
        // Skip labels for simplicity in this implementation
        
        Ok(FeatureVector::new(values, object_type, encoding_strategy))
    }
    
    #[cfg(feature = "serde")]
    fn deserialize_json(&self, data: &[u8]) -> FeatureResult<FeatureVector> {
        let json_str = String::from_utf8(data.to_vec())
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        
        serde_json::from_str(&json_str)
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))
    }
    
    #[cfg(not(feature = "serde"))]
    fn deserialize_json(&self, data: &[u8]) -> FeatureResult<FeatureVector> {
        // Simple JSON deserialization - in practice you'd want a proper parser
        let json_str = String::from_utf8(data.to_vec())
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        
        // Extract values array (very basic parsing)
        let values_start = json_str.find("\"values\":[")
            .ok_or_else(|| FeatureError::DeserializationFailed(
                "Values array not found".to_string()
            ))? + 10;
        let values_end = json_str[values_start..].find(']')
            .ok_or_else(|| FeatureError::DeserializationFailed(
                "Values array end not found".to_string()
            ))? + values_start;
        
        let values_str = &json_str[values_start..values_end];
        let values: Vec<f64> = values_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        
        Ok(FeatureVector::new(
            values,
            "json_deserialized".to_string(),
            "json".to_string(),
        ))
    }
    
    fn deserialize_messagepack(&self, _data: &[u8]) -> FeatureResult<FeatureVector> {
        Err(FeatureError::DeserializationFailed(
            "MessagePack deserialization not implemented".to_string()
        ))
    }
    
    fn deserialize_custom(&self, data: &[u8]) -> FeatureResult<FeatureVector> {
        let mut offset = 0;
        
        // Check magic header
        if data.len() < 6 || &data[0..4] != b"RUVF" {
            return Err(FeatureError::DeserializationFailed(
                "Invalid custom format header".to_string()
            ));
        }
        offset += 4;
        
        // Check version
        let version = data[offset];
        if version != 1 {
            return Err(FeatureError::DeserializationFailed(
                format!("Unsupported format version: {}", version)
            ));
        }
        offset += 1;
        
        // Read flags
        let flags = data[offset];
        offset += 1;
        
        // Read dimension
        let (dimension, bytes_read) = self.read_varint(&data[offset..])?;
        offset += bytes_read;
        
        // Read values
        let (values, bytes_read) = self.read_optimized_values(&data[offset..], dimension as usize)?;
        offset += bytes_read;
        
        // Read metadata if present
        let (object_type, encoding_strategy) = if flags & 0x01 != 0 {
            let (obj_type, bytes_read) = self.read_string(&data[offset..])?;
            offset += bytes_read;
            
            let (enc_strategy, bytes_read) = self.read_string(&data[offset..])?;
            offset += bytes_read;
            
            // Skip timestamp
            offset += 8;
            
            (obj_type, enc_strategy)
        } else {
            ("custom_deserialized".to_string(), "custom".to_string())
        };
        
        // Read labels if present
        let labels = if flags & 0x02 != 0 {
            let (label_count, bytes_read) = self.read_varint(&data[offset..])?;
            offset += bytes_read;
            
            let mut labels = Vec::with_capacity(label_count as usize);
            for _ in 0..label_count {
                let (label, bytes_read) = self.read_string(&data[offset..])?;
                offset += bytes_read;
                labels.push(label);
            }
            Some(labels)
        } else {
            None
        };
        
        Ok(FeatureVector::new(
            values,
            object_type,
            encoding_strategy,
        ).with_labels(labels.unwrap_or_default()))
    }
    
    // Compression methods
    
    #[cfg(feature = "compression")]
    fn compress(&self, data: &[u8]) -> FeatureResult<Vec<u8>> {
        match self.config.compression {
            CompressionType::Gzip => {
                use flate2::{Compression, write::GzEncoder};
                use std::io::Write;
                
                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    Compression::new(self.config.compression_level)
                );
                encoder.write_all(data)
                    .map_err(|e| FeatureError::SerializationFailed(e.to_string()))?;
                encoder.finish()
                    .map_err(|e| FeatureError::SerializationFailed(e.to_string()))
            }
            CompressionType::Zlib => {
                use flate2::{Compression, write::ZlibEncoder};
                use std::io::Write;
                
                let mut encoder = ZlibEncoder::new(
                    Vec::new(),
                    Compression::new(self.config.compression_level)
                );
                encoder.write_all(data)
                    .map_err(|e| FeatureError::SerializationFailed(e.to_string()))?;
                encoder.finish()
                    .map_err(|e| FeatureError::SerializationFailed(e.to_string()))
            }
            _ => Err(FeatureError::SerializationFailed(
                "Compression type not supported".to_string()
            ))
        }
    }
    
    #[cfg(not(feature = "compression"))]
    fn compress(&self, data: &[u8]) -> FeatureResult<Vec<u8>> {
        // Simple run-length encoding as fallback
        let mut compressed = Vec::new();
        if data.is_empty() {
            return Ok(compressed);
        }
        
        let mut current_byte = data[0];
        let mut count = 1u8;
        
        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }
        
        // Add the last run
        compressed.push(count);
        compressed.push(current_byte);
        
        Ok(compressed)
    }
    
    #[cfg(feature = "compression")]
    fn decompress(&self, data: &[u8]) -> FeatureResult<Vec<u8>> {
        match self.config.compression {
            CompressionType::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;
                
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)
                    .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
                Ok(decompressed)
            }
            CompressionType::Zlib => {
                use flate2::read::ZlibDecoder;
                use std::io::Read;
                
                let mut decoder = ZlibDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)
                    .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
                Ok(decompressed)
            }
            _ => Err(FeatureError::DeserializationFailed(
                "Compression type not supported".to_string()
            ))
        }
    }
    
    #[cfg(not(feature = "compression"))]
    fn decompress(&self, data: &[u8]) -> FeatureResult<Vec<u8>> {
        // Simple run-length decoding
        let mut decompressed = Vec::new();
        
        let mut i = 0;
        while i + 1 < data.len() {
            let count = data[i];
            let byte_value = data[i + 1];
            
            for _ in 0..count {
                decompressed.push(byte_value);
            }
            
            i += 2;
        }
        
        Ok(decompressed)
    }
    
    // Helper methods for custom format
    
    fn write_varint(&self, data: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            data.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        data.push(value as u8);
    }
    
    fn read_varint(&self, data: &[u8]) -> FeatureResult<(u64, usize)> {
        let mut value = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;
        
        for &byte in data {
            bytes_read += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(FeatureError::DeserializationFailed(
                    "Varint too large".to_string()
                ));
            }
        }
        
        Ok((value, bytes_read))
    }
    
    fn write_string(&self, data: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        self.write_varint(data, bytes.len() as u64);
        data.extend_from_slice(bytes);
    }
    
    fn read_string(&self, data: &[u8]) -> FeatureResult<(String, usize)> {
        let (length, varint_bytes) = self.read_varint(data)?;
        let string_bytes = length as usize;
        
        if data.len() < varint_bytes + string_bytes {
            return Err(FeatureError::DeserializationFailed(
                "Insufficient data for string".to_string()
            ));
        }
        
        let string_data = &data[varint_bytes..varint_bytes + string_bytes];
        let string = String::from_utf8(string_data.to_vec())
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        
        Ok((string, varint_bytes + string_bytes))
    }
    
    fn write_optimized_values(&self, data: &mut Vec<u8>, values: &[f64]) -> FeatureResult<()> {
        // Analyze values to choose optimal encoding
        let mut all_integers = true;
        let mut max_abs = 0.0;
        
        for &value in values {
            if value.fract() != 0.0 {
                all_integers = false;
            }
            max_abs = max_abs.max(value.abs());
        }
        
        if all_integers && max_abs <= i32::MAX as f64 {
            // Use integer encoding
            data.push(1); // encoding type
            for &value in values {
                data.extend_from_slice(&(value as i32).to_le_bytes());
            }
        } else {
            // Use full float encoding
            data.push(2); // encoding type
            for &value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        Ok(())
    }
    
    fn read_optimized_values(&self, data: &[u8], count: usize) -> FeatureResult<(Vec<f64>, usize)> {
        if data.is_empty() {
            return Err(FeatureError::DeserializationFailed(
                "No encoding type specified".to_string()
            ));
        }
        
        let encoding_type = data[0];
        let mut offset = 1;
        let mut values = Vec::with_capacity(count);
        
        match encoding_type {
            1 => {
                // Integer encoding
                let required_bytes = count * 4;
                if data.len() < offset + required_bytes {
                    return Err(FeatureError::DeserializationFailed(
                        "Insufficient data for integer values".to_string()
                    ));
                }
                
                for i in 0..count {
                    let int_bytes = &data[offset + i * 4..offset + (i + 1) * 4];
                    let int_value = i32::from_le_bytes(int_bytes.try_into().unwrap());
                    values.push(int_value as f64);
                }
                offset += required_bytes;
            }
            2 => {
                // Float encoding
                let required_bytes = count * 8;
                if data.len() < offset + required_bytes {
                    return Err(FeatureError::DeserializationFailed(
                        "Insufficient data for float values".to_string()
                    ));
                }
                
                for i in 0..count {
                    let float_bytes = &data[offset + i * 8..offset + (i + 1) * 8];
                    let float_value = f64::from_le_bytes(float_bytes.try_into().unwrap());
                    values.push(float_value);
                }
                offset += required_bytes;
            }
            _ => {
                return Err(FeatureError::DeserializationFailed(
                    format!("Unknown encoding type: {}", encoding_type)
                ));
            }
        }
        
        Ok((values, offset))
    }
    
    fn validate_deserialized(&self, features: &FeatureVector) -> FeatureResult<()> {
        // Basic integrity checks
        if features.values.len() != features.dimension {
            return Err(FeatureError::ValidationFailed {
                message: "Dimension mismatch after deserialization".to_string(),
            });
        }
        
        // Check for invalid floating-point values
        for (i, &value) in features.values.iter().enumerate() {
            if value.is_nan() {
                return Err(FeatureError::ValidationFailed {
                    message: format!("NaN value at index {} after deserialization", i),
                });
            }
            if value.is_infinite() {
                return Err(FeatureError::ValidationFailed {
                    message: format!("Infinite value at index {} after deserialization", i),
                });
            }
        }
        
        Ok(())
    }
}

impl Default for FeatureSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_feature_vector() -> FeatureVector {
        FeatureVector::new(
            vec![1.0, 2.5, -3.0, 4.7],
            "test".to_string(),
            "test_encoder".to_string(),
        ).with_labels(vec![
            "feature_1".to_string(),
            "feature_2".to_string(),
            "feature_3".to_string(),
            "feature_4".to_string(),
        ])
    }
    
    #[test]
    fn test_serialization_config_default() {
        let config = SerializationConfig::default();
        assert_eq!(config.format, SerializationFormat::Binary);
        assert_eq!(config.compression, CompressionType::Gzip);
        assert_eq!(config.compression_level, 6);
        assert!(config.include_metadata);
        assert!(config.include_labels);
    }
    
    #[test]
    fn test_binary_serialization_roundtrip() {
        let serializer = FeatureSerializer::new();
        let original = create_test_feature_vector();
        
        let serialized = serializer.serialize(&original).unwrap();
        let deserialized = serializer.deserialize(&serialized).unwrap();
        
        assert_eq!(original.values, deserialized.values);
        assert_eq!(original.dimension, deserialized.dimension);
        assert_eq!(original.metadata.object_type, deserialized.metadata.object_type);
    }
    
    #[test]
    fn test_custom_format_serialization() {
        let config = SerializationConfig {
            format: SerializationFormat::Custom,
            compression: CompressionType::None,
            ..Default::default()
        };
        let serializer = FeatureSerializer::with_config(config);
        let original = create_test_feature_vector();
        
        let serialized = serializer.serialize(&original).unwrap();
        let deserialized = serializer.deserialize(&serialized).unwrap();
        
        assert_eq!(original.values, deserialized.values);
        assert_eq!(original.dimension, deserialized.dimension);
    }
    
    #[test]
    fn test_batch_serialization() {
        let serializer = FeatureSerializer::new();
        let features = vec![
            create_test_feature_vector(),
            create_test_feature_vector(),
        ];
        
        let serialized = serializer.serialize_batch(&features).unwrap();
        let deserialized = serializer.deserialize_batch(&serialized).unwrap();
        
        assert_eq!(features.len(), deserialized.len());
        for (orig, deser) in features.iter().zip(deserialized.iter()) {
            assert_eq!(orig.values, deser.values);
        }
    }
    
    #[test]
    fn test_compression_types() {
        for compression in [CompressionType::None, CompressionType::Gzip] {
            let config = SerializationConfig {
                compression,
                ..Default::default()
            };
            let serializer = FeatureSerializer::with_config(config);
            let original = create_test_feature_vector();
            
            let serialized = serializer.serialize(&original).unwrap();
            let deserialized = serializer.deserialize(&serialized).unwrap();
            
            assert_eq!(original.values, deserialized.values);
        }
    }
}