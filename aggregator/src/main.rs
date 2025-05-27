use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde_json::{Deserializer, Value};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

fn main() -> std::io::Result<()> {
    let output_path = "./../hackaton/datasets/merged/merged_dataset_v2.json.gz";
    let input_paths = [
        "./../hackaton/datasets/A/train.json.gz",
        "./../hackaton/datasets/B/train.json.gz",
        "./../hackaton/datasets/C/train.json.gz",
        "./../hackaton/datasets/D/train.json.gz",
    ];

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(GzEncoder::new(output_file, Compression::default()));

    writeln!(writer, "[")?;

    let mut first = true;
    for path in input_paths {
        println!("Reading: {}", path);
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);

        let stream = Deserializer::from_reader(reader).into_iter::<Value>();
        for value in stream {
            if let Ok(json_value) = value {
                if !first {
                    writeln!(writer, ",")?;
                } else {
                    first = false;
                }
                serde_json::to_writer(&mut writer, &json_value)?;
            }
        }
    }

    writeln!(writer, "\n]")?;
    writer.flush()?;

    println!("✅ Merged {} files into {}", input_paths.len(), output_path);
    Ok(())
}
