use futures::{AsyncBufRead, AsyncBufReadExt};
use futures_lite::{pin, StreamExt};
use std::collections::HashMap;
use std::num::NonZero;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseHeaderError {
    #[error("The provided type is not supported by the NRRD format: \"{0}\"")]
    TypeParseError(String),

    #[error("The provided encoding is not supported by the NRRD format: \"{0}\"")]
    EncodingParseError(String),

    #[error("The provided endian type is not supported by the NRRD format: \"{0}\"")]
    EndianParseError(String),

    #[error("The provided space type is not supported by the NRRD format: \"{0}\"")]
    SpaceParseError(String),

    #[error("The provided centering is not supported by the NRRD format: \"{0}\"")]
    CenterParseError(String),

    #[error("The provided kind is not supported by the NRRD format: \"{0}\"")]
    KindParseError(String),

    #[error("The provided input file is not supported by the NRRD format: \"{0}\"")]
    FileParseError(String),

    #[error("Failed reading from async file stream: \"{0}\"")]
    FileReadAsyncError(#[from] futures::io::Error),

    #[error("Failed parsing int: \"{0}\"")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("Failed parsing float: \"{0}\"")]
    ParseFloatError(#[from] std::num::ParseFloatError),

    #[error("Expected non zero value for field: \"{0}\"")]
    ExpectedNonZeroError(String),
}

#[derive(Debug, PartialEq)]
pub enum NRRDType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Block,
}

impl FromStr for NRRDType {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "signed char" | "int8" | "int8_t" => Ok(NRRDType::I8),
            "uchar" | "unsigned char" | "uint8" | "uint8_t" => Ok(NRRDType::U8),
            "short" | "short int" | "signed short" | "signed short int" | "int16" | "int16_t" => {
                Ok(NRRDType::I16)
            }
            "ushort" | "unsigned short" | "unsigned short int" | "uint16" | "uint16_t" => {
                Ok(NRRDType::U16)
            }
            "int" | "signed int" | "int32" | "int32_t" => Ok(NRRDType::I32),
            "uint" | "unsigned int" | "uint32" | "uint32_t" => Ok(NRRDType::U32),
            "longlong"
            | "long long"
            | "long long int"
            | "signed long long"
            | "signed long long int"
            | "int64"
            | "int64_t" => Ok(NRRDType::I64),
            "ulonglong"
            | "unsigned long long"
            | "unsigned long long int"
            | "uint64"
            | "uint64_t" => Ok(NRRDType::U64),
            "float" => Ok(NRRDType::F32),
            "double" => Ok(NRRDType::F64),
            "block" => Ok(NRRDType::Block),
            _ => Err(ParseHeaderError::TypeParseError(name.to_string())),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NRRDEncoding {
    Raw,
    Ascii,
    Hex,
    GZip,
    BZip2,
}

impl FromStr for NRRDEncoding {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "raw" => Ok(NRRDEncoding::Raw),
            "txt" | "text" | "ascii" => Ok(NRRDEncoding::Ascii),
            "hex" => Ok(NRRDEncoding::Hex),
            "gz" | "gzip" => Ok(NRRDEncoding::GZip),
            "bz2" | "bzip2" => Ok(NRRDEncoding::BZip2),
            _ => Err(ParseHeaderError::EncodingParseError(name.to_string())),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NRRDEndian {
    LittleEndian,
    BigEndian,
}

impl FromStr for NRRDEndian {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "little" => Ok(NRRDEndian::LittleEndian),
            "big" => Ok(NRRDEndian::BigEndian),
            _ => Err(ParseHeaderError::EndianParseError(name.to_string())),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NRRDCentering {
    Cell,
    Node,
    None,
}

impl FromStr for NRRDCentering {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "cell" => Ok(NRRDCentering::Cell),
            "node" => Ok(NRRDCentering::Node),
            "???" | "None" => Ok(NRRDCentering::None),
            _ => Err(ParseHeaderError::CenterParseError(name.to_string())),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NRRDSpace {
    RightAnteriorSuperior,
    LeftAnteriorSuperior,
    LeftPosteriorSuperior,
    RightAnteriorSuperiorTime,
    LeftAnteriorSuperiorTime,
    LeftPosteriorSuperiorTime,
    ScannerXYZ,
    ScannerXYZTime,
    RightHanded3d,
    LeftHanded3d,
    RightHanded3dTime,
    LeftHanded3dTime,
}

impl FromStr for NRRDSpace {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "right-anterior-superior" | "RAS" => Ok(NRRDSpace::RightAnteriorSuperior),
            "left-anterior-superior" | "LAS" => Ok(NRRDSpace::LeftAnteriorSuperior),
            "left-posterior-superior" | "LPS" => Ok(NRRDSpace::LeftPosteriorSuperior),
            "right-anterior-superior-time" | "RAST" => Ok(NRRDSpace::RightAnteriorSuperiorTime),
            "left-anterior-superior-time" | "LAST" => Ok(NRRDSpace::LeftAnteriorSuperiorTime),
            "left-posterior-superior-time" | "LPST" => Ok(NRRDSpace::LeftPosteriorSuperiorTime),
            "scanner-xyz" => Ok(NRRDSpace::ScannerXYZ),
            "scanner-xyz-time" => Ok(NRRDSpace::ScannerXYZTime),
            "3D-right-handed" => Ok(NRRDSpace::RightHanded3d),
            "3D-left-handed" => Ok(NRRDSpace::LeftHanded3d),
            "3D-right-handed-time" => Ok(NRRDSpace::RightHanded3dTime),
            "3D-left-handed-time" => Ok(NRRDSpace::LeftHanded3dTime),
            _ => Err(ParseHeaderError::SpaceParseError(name.to_string())),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NRRDKind {
    Domain,
    Space,
    Time,
    List,
    Point,
    Vector,
    CovariantVector,
    Normal,
    Stub,
    Scalar,
    Complex,
    Vector2,
    Color3,
    ColorRGB,
    ColorHSV,
    ColorXYZ,
    Color4,
    ColorRGBA,
    Vector3,
    Gradient3,
    Normal3,
    Vector4,
    Quaternion,
    SymmetricMatrix2d,
    MaskedSymmetricMatric2d,
    Matrix2d,
    MaskedMatrix2d,
    SymmetricMatrix3d,
    MaskedSymmetricMatrix3d,
    Matrix3d,
    MaskedMatrix3d,
    None,
}

impl FromStr for NRRDKind {
    type Err = ParseHeaderError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "domain" => Ok(NRRDKind::Domain),
            "space" => Ok(NRRDKind::Space),
            "time" => Ok(NRRDKind::Time),
            "list" => Ok(NRRDKind::List),
            "point" => Ok(NRRDKind::Point),
            "vector" => Ok(NRRDKind::Vector),
            "covariant-vector" => Ok(NRRDKind::CovariantVector),
            "normal" => Ok(NRRDKind::Normal),
            "stub" => Ok(NRRDKind::Stub),
            "scalar" => Ok(NRRDKind::Scalar),
            "complex" => Ok(NRRDKind::Complex),
            "2-vector" => Ok(NRRDKind::Vector2),
            "3-color" => Ok(NRRDKind::Color3),
            "RGB-color" => Ok(NRRDKind::ColorRGB),
            "HSV-color" => Ok(NRRDKind::ColorHSV),
            "XYZ-color" => Ok(NRRDKind::ColorXYZ),
            "4-color" => Ok(NRRDKind::Color4),
            "RGBA-color" => Ok(NRRDKind::ColorRGBA),
            "3-vector" => Ok(NRRDKind::Vector3),
            "3-gradient" => Ok(NRRDKind::Gradient3),
            "3-normal" => Ok(NRRDKind::Normal3),
            "4-vector" => Ok(NRRDKind::Vector4),
            "quaternion" => Ok(NRRDKind::Quaternion),
            "2D-symmetric-matrix" => Ok(NRRDKind::SymmetricMatrix2d),
            "2D-masked-symmetric-matrix" => Ok(NRRDKind::MaskedSymmetricMatric2d),
            "2d-matrix" => Ok(NRRDKind::Matrix2d),
            "2d-masked-matrix" => Ok(NRRDKind::MaskedMatrix2d),
            "3D-symmetric-matrix" => Ok(NRRDKind::SymmetricMatrix3d),
            "3D-masked-symmetric-matrix" => Ok(NRRDKind::MaskedSymmetricMatrix3d),
            "3D-matrix" => Ok(NRRDKind::Matrix3d),
            "3D-masked-matrix" => Ok(NRRDKind::MaskedMatrix3d),
            "???" | "none" => Ok(NRRDKind::None),
            _ => Err(ParseHeaderError::KindParseError(name.to_string())),
        }
    }
}

/// Contents of a NRRD Header file as defined by https://teem.sourceforge.net/nrrd/format.html
#[derive(Debug)]
pub struct NRRDHeader {
    // Basic Fields
    pub dimension: NonZero<usize>,
    pub data_type: NRRDType,
    pub block_size: Option<NonZero<usize>>,
    pub encoding: NRRDEncoding,
    pub endian: Option<NRRDEndian>,
    pub content: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub old_min: Option<f64>,
    pub old_max: Option<f64>,
    pub data_file: Option<String>,
    pub line_skip: Option<usize>,
    pub byte_skip: Option<usize>,
    pub sample_units: Option<String>,

    // Per axis fields
    pub sizes: Vec<NonZero<usize>>,
    pub spacings: Option<Vec<f64>>,
    pub thicknesses: Option<Vec<f64>>,
    pub axis_mins: Option<Vec<f64>>,
    pub axis_maxs: Option<Vec<f64>>,
    pub centers: Option<Vec<NRRDCentering>>,
    pub labels: Option<Vec<String>>,
    pub units: Option<Vec<String>>,
    pub kinds: Option<Vec<NRRDKind>>,

    // Space and orientation information
    pub space: Option<NRRDSpace>,
    pub space_dimension: Option<NonZero<usize>>,
    pub space_units: Option<Vec<String>>,
    pub space_origin: Option<Vec<f64>>,
    pub space_directions: Option<Vec<Option<Vec<f64>>>>,
    pub measurement_frame: Option<Vec<f64>>,
}

impl NRRDHeader {
    fn get_either_field(map: &mut HashMap<String, String>, fields: &[&str]) -> Option<String> {
        for field in fields {
            if let Some(value) = map.remove(*field) {
                return Some(value.to_string());
            }
        }

        None
    }

    fn parse_string_list_field(
        map: &mut HashMap<String, String>,
        fields: &[&str],
    ) -> Result<Option<Vec<String>>, ParseHeaderError> {
        let field = fields[0];
        let value = Self::get_either_field(map, fields);
        if let Some(s) = value {
            let mut in_string = false;
            let mut start = 0usize;
            let mut strings = Vec::new();

            let mut prev_char = '\0';

            for (i, c) in s.chars().enumerate() {
                if c == '"' && !in_string {
                    in_string = true;
                    start = i;

                    // terminate if field is not proceeded by space
                    if i > 0 && prev_char != ' ' {
                        return Err(ParseHeaderError::FileParseError(format!(
                            "Failed to parse {field} field: unexpected quote found at position {i}"
                        )));
                    }
                } else if c == '"' && in_string && i > 0 && prev_char != '\\' {
                    in_string = false;
                    strings.push(s[start + 1..i].to_string());
                }

                prev_char = c;
            }

            if strings.is_empty() {
                return Err(ParseHeaderError::FileParseError(format!(
                    "Failed to parse {field} field: no valid strings found"
                )));
            }

            if in_string {
                return Err(ParseHeaderError::FileParseError(format!(
                    "Failed to parse {field} field: unmatched quote found"
                )));
            }

            Ok(Some(strings))
        } else {
            Ok(None)
        }
    }

    fn parse_vector(value: &str) -> Result<Vec<f64>, ParseHeaderError> {
        if value.trim().starts_with("(") && value.trim().ends_with(")") {
            let components = value[1..value.len() - 1].split(",");
            components
                .map(|c| {
                    c.trim()
                        .parse::<f64>()
                        .map_err(|e| ParseHeaderError::ParseFloatError(e))
                })
                .collect()
        } else {
            Err(ParseHeaderError::FileParseError(format!(
                "Failed to parse vector field: expected value to be enclosed in parentheses, got: {value}"
            )))
        }
    }
    fn from_fields(
        mut field_desc_pairs: HashMap<String, String>,
        _key_value_pairs: HashMap<String, String>,
    ) -> Result<Self, ParseHeaderError> {
        let dimension =
            field_desc_pairs
                .remove("dimension")
                .ok_or(ParseHeaderError::FileParseError(
                    "NRRD file is missing required field: dimension".to_string(),
                ))?;
        let dimension = NonZero::new(dimension.parse::<usize>()?).ok_or(
            ParseHeaderError::ExpectedNonZeroError("dimension".to_string()),
        )?;

        let data_type = field_desc_pairs
            .remove("type")
            .ok_or(ParseHeaderError::FileParseError(
                "NRRD file is missing required field: type".to_string(),
            ))?;
        let data_type = data_type.parse()?;

        let block_size =
            Self::get_either_field(&mut field_desc_pairs, &["block size", "blocksize"])
                .map(|s| {
                    NonZero::new(s.parse::<usize>()?).ok_or(ParseHeaderError::ExpectedNonZeroError(
                        "block size".to_string(),
                    ))
                })
                .transpose()?;

        let encoding = field_desc_pairs
            .remove("encoding")
            .ok_or(ParseHeaderError::FileParseError(
                "NRRD file is missing required field: encoding".to_string(),
            ))?
            .parse()?;

        let endian = field_desc_pairs
            .remove("endian")
            .map(|s| s.parse())
            .transpose()?;

        let content = field_desc_pairs.remove("content");

        let min = field_desc_pairs
            .remove("min")
            .map(|s| s.parse::<f64>())
            .transpose()?;

        let max = field_desc_pairs
            .remove("max")
            .map(|s| s.parse::<f64>())
            .transpose()?;

        let old_min = Self::get_either_field(&mut field_desc_pairs, &["old min", "oldmin"])
            .map(|s| s.parse::<f64>())
            .transpose()?;

        let old_max = Self::get_either_field(&mut field_desc_pairs, &["old max", "oldmax"])
            .map(|s| s.parse::<f64>())
            .transpose()?;

        // TODO more complicated parsing
        let data_file = Self::get_either_field(&mut field_desc_pairs, &["data file", "datafile"]);

        let line_skip = Self::get_either_field(&mut field_desc_pairs, &["line skip", "lineskip"])
            .map(|s| s.parse::<usize>())
            .transpose()?;

        let byte_skip = Self::get_either_field(&mut field_desc_pairs, &["byte skip", "byteskip"])
            .map(|s| s.parse::<usize>())
            .transpose()?;

        let sample_units =
            Self::get_either_field(&mut field_desc_pairs, &["sample units", "sampleunits"]);

        let sizes = field_desc_pairs
            .remove("sizes")
            .ok_or(ParseHeaderError::FileParseError(
                "NRRD file is missing required field: sizes".to_string(),
            ))?;
        let sizes = sizes
            .split(" ")
            .map(|s| {
                NonZero::new(s.parse::<usize>()?)
                    .ok_or(ParseHeaderError::ExpectedNonZeroError("sizes".to_string()))
            })
            .collect::<Result<Vec<_>, ParseHeaderError>>()?;

        let spacings = field_desc_pairs
            .remove("spacings")
            .map(|s| s.split(' ').map(|s| s.parse::<f64>()).collect())
            .transpose()?;

        let thicknesses = field_desc_pairs
            .remove("thicknesses")
            .map(|s| s.split(' ').map(|s| s.parse::<f64>()).collect())
            .transpose()?;

        let axis_mins = Self::get_either_field(&mut field_desc_pairs, &["axis mins", "axismins"])
            .map(|s| s.split(' ').map(|s| s.parse::<f64>()).collect())
            .transpose()?;

        let axis_maxs = Self::get_either_field(&mut field_desc_pairs, &["axis maxs", "axismaxs"])
            .map(|s| s.split(' ').map(|s| s.parse::<f64>()).collect())
            .transpose()?;

        let centers = Self::get_either_field(&mut field_desc_pairs, &["centers", "centerings"])
            .map(|s| s.split(' ').map(|s| s.parse()).collect())
            .transpose()?;

        let labels = Self::parse_string_list_field(&mut field_desc_pairs, &["labels"])?;
        let units = Self::parse_string_list_field(&mut field_desc_pairs, &["units"])?;

        let kinds = field_desc_pairs
            .remove("kinds")
            .map(|s| {
                s.split(" ")
                    .map(|v| v.parse())
                    .collect::<Result<Vec<_>, ParseHeaderError>>()
            })
            .transpose()?;

        let space = field_desc_pairs
            .remove("space")
            .map(|s| s.parse())
            .transpose()?;

        let space_dimension = field_desc_pairs
            .remove("space dimension")
            .map(|s| {
                NonZero::new(s.parse::<usize>()?).ok_or(ParseHeaderError::ExpectedNonZeroError(
                    "space dimension".to_string(),
                ))
            })
            .transpose()?;

        let space_units = Self::parse_string_list_field(&mut field_desc_pairs, &["space units"])?;

        let space_origin = field_desc_pairs
            .remove("space origin")
            .map(|s| Self::parse_vector(s.as_str()))
            .transpose()?;

        let space_directions: Option<Vec<Option<Vec<f64>>>> = field_desc_pairs
            .remove("space directions")
            .map(|s| {
                // map option inner
                s.split(' ')
                    .map(|s| {
                        // map each vector
                        if s.trim() == "none" {
                            Ok(None)
                        } else {
                            Self::parse_vector(s).map(Some)
                        }
                    })
                    .collect() // into array of vector | None
            })
            .transpose()?;

        let measurement_frame = field_desc_pairs
            .remove("measurement frame")
            .map(|s| -> Result<Vec<f64>, ParseHeaderError> {
                let vs: Vec<Vec<f64>> = s
                    .trim()
                    .split(' ')
                    .map(|s| Self::parse_vector(s))
                    .collect::<Result<Vec<_>, ParseHeaderError>>()?;
                Ok(vs.into_iter().flatten().collect())
            })
            .transpose()?;

        let _ = field_desc_pairs.remove("number"); // unused, but defined

        let unrecognized_fields = field_desc_pairs.keys().collect::<Vec<_>>();
        if !unrecognized_fields.is_empty() {
            return Err(ParseHeaderError::FileParseError(format!(
                "NRRD file contains unrecognized fields: {:?}",
                unrecognized_fields,
            )));
        }

        Ok(Self {
            dimension,
            data_type,
            block_size,
            encoding,
            endian,
            content,
            min,
            max,
            old_min,
            old_max,
            data_file,
            line_skip,
            byte_skip,
            sample_units,
            sizes,
            spacings,
            thicknesses,
            axis_mins,
            axis_maxs,
            centers,
            labels,
            units,
            kinds,
            space,
            space_dimension,
            space_units,
            space_origin,
            space_directions,
            measurement_frame,
        })
    }

    pub async fn from_buf_async<B: AsyncBufRead>(
        reader: B,
    ) -> Result<NRRDHeader, ParseHeaderError> {
        pin!(reader);
        let mut lines = reader.lines();

        if let Some(line) = lines.next().await {
            Self::parse_nrrd_magic(line?.trim())?;
        } else {
            return Err(ParseHeaderError::FileParseError(
                "NRRD file is empty".to_string(),
            ));
        }

        let mut field_desc_pairs = HashMap::new();
        let mut key_value_pairs = HashMap::new();

        while let Some(line) = lines.next().await {
            let line = line?;

            if line.trim().is_empty() {
                // empty line separates header from data
                break;
            }

            if line.starts_with("#") {
                // comment
                continue;
            }

            let split: Vec<&str> = line.split(": ").collect();
            if split.len() == 2 {
                field_desc_pairs.insert(split[0].to_string(), split[1].trim().to_string());
                continue;
            }

            let split: Vec<&str> = line.split(":=").collect();
            if split.len() == 2 {
                key_value_pairs.insert(split[0].to_string(), split[1].trim().to_string());
            }
        }

        Self::from_fields(field_desc_pairs, key_value_pairs)
    }

    fn parse_nrrd_magic(magic: &str) -> Result<u8, ParseHeaderError> {
        if !magic.starts_with("NRRD000") || magic.len() < 8 {
            return Err(ParseHeaderError::FileParseError(
                "NRRD file does not start with a valid NRRD magic number".to_string(),
            ));
        }

        let version: u8 = magic[4..8].parse().map_err(|_| {
            ParseHeaderError::FileParseError(format!(
                "NRRD file has an invalid version number: {magic}"
            ))
        })?;
        if version < 2 || version > 5 {
            return Err(ParseHeaderError::FileParseError(format!(
                "NRRD version {magic} is not supported"
            )));
        }

        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_magic() {
        assert_eq!(NRRDHeader::parse_nrrd_magic("NRRD0002").unwrap(), 2);
        assert_eq!(NRRDHeader::parse_nrrd_magic("NRRD0003").unwrap(), 3);
        assert_eq!(NRRDHeader::parse_nrrd_magic("NRRD0004").unwrap(), 4);
        assert_eq!(NRRDHeader::parse_nrrd_magic("NRRD0005").unwrap(), 5);

        assert!(NRRDHeader::parse_nrrd_magic("NRRD0000").is_err());
        assert!(NRRDHeader::parse_nrrd_magic("NRRD0001").is_err());
        assert!(NRRDHeader::parse_nrrd_magic("NRRD0006").is_err());
        assert!(NRRDHeader::parse_nrrd_magic("NRRD000").is_err());
        assert!(NRRDHeader::parse_nrrd_magic("Weird String").is_err());
    }

    #[test]
    fn parse_detached() {
        let data = r#"NRRD0004
# Complete NRRD file format specification at:
# http://teem.sourceforge.net/nrrd/format.html
type: uint8
dimension: 3
space: left-posterior-superior
sizes: 256 256 256
space directions: (1,0,0) (0,1,0) (0,0,1)
kinds: domain domain domain
endian: little
encoding: raw
space origin: (-128.0,-128.0,-128.0)
data file: aneurism_256x256x256_uint8.raw
"#;
        let reader = futures::io::Cursor::new(data);
        let header = NRRDHeader::from_buf_async(reader);
        let header = pollster::block_on(header);
        let header = header.unwrap();
        assert_eq!(header.data_type, NRRDType::U8);
        assert_eq!(header.dimension, NonZero::new(3).unwrap());
        assert_eq!(header.space, Some(NRRDSpace::LeftPosteriorSuperior));
        let sizes: Vec<_> = header.sizes.iter().map(|s| s.get()).collect();
        assert_eq!(sizes, vec![256, 256, 256]);
        assert_eq!(
            header.space_directions,
            Some(vec![
                Some(vec![1.0, 0.0, 0.0,]),
                Some(vec![0.0, 1.0, 0.0,]),
                Some(vec![0.0, 0.0, 1.0,]),
            ])
        );
        assert_eq!(header.endian, Some(NRRDEndian::LittleEndian));
        assert_eq!(header.encoding, NRRDEncoding::Raw);
        assert_eq!(header.space_origin, Some(vec![-128.0, -128.0, -128.0]));
        assert_eq!(
            header.data_file,
            Some("aneurism_256x256x256_uint8.raw".to_string())
        );
    }
}
