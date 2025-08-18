#[cfg(target_arch = "wasm32")]
use std::io::Cursor;
use std::io::Read;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

#[cfg(target_arch = "wasm32")]
pub struct FileWriter {
    writer: Cursor<Vec<u8>>,
    path: String,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct FileWriter {
    writer: std::io::BufWriter<std::fs::File>,
}

impl Deref for FileWriter {
    #[cfg(target_arch = "wasm32")]
    type Target = Cursor<Vec<u8>>;
    #[cfg(not(target_arch = "wasm32"))]
    type Target = std::io::BufWriter<std::fs::File>;

    fn deref(&self) -> &Self::Target {
        #[cfg(target_arch = "wasm32")]
        {
            &self.writer
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            &self.writer
        }
    }
}

impl DerefMut for FileWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        #[cfg(target_arch = "wasm32")]
        {
            &mut self.writer
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            &mut self.writer
        }
    }
}

impl FileWriter {
    #[cfg(target_arch = "wasm32")]
    pub fn close(&self) {
        use web_sys::js_sys;
        use web_sys::wasm_bindgen::JsCast;
        use web_sys::{Blob, BlobPropertyBag, Document, HtmlAnchorElement, Url, Window};

        let uint8_array = js_sys::Uint8Array::from(self.writer.get_ref().as_slice());
        let array_for_blob = js_sys::Array::new();
        array_for_blob.push(&uint8_array.into());
        let blob =
            Blob::new_with_u8_array_sequence(&array_for_blob).expect("failed to create blob");
        let window: Window = web_sys::window().expect("window not found");
        let url = Url::create_object_url_with_blob(&blob).expect("failed to create object URL");
        let document: Document = window.document().expect("document not found");
        let a: HtmlAnchorElement = document
            .create_element("a")
            .expect("failed to create anchor element")
            .dyn_into::<HtmlAnchorElement>()
            .expect("failed to cast to HtmlAnchorElement");
        a.set_href(&url);
        a.set_download(&self.path);
        let body = document.body().expect("document body not found");
        body.append_child(&a)
            .expect("failed to append anchor element to body");
        a.click();
        body.remove_child(&a)
            .expect("failed to remove anchor element");
        Url::revoke_object_url(&url).expect("failed to revoke object URL");
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn close(&mut self) {
        use std::io::Write;

        self.writer.flush().expect("Failed to flush writer");
    }
}

impl Drop for FileWriter {
    fn drop(&mut self) {
        self.close();
    }
}

pub struct Filesystem {
    out_dir: PathBuf,
}

impl Filesystem {
    pub fn new(prefix: Option<String>) -> Self {
        let time = chrono::Local::now();
        let prefix = prefix.unwrap_or("aurora".to_string());
        let dirname = format!("out/{prefix}_{}", time.format("%Y-%m-%d_%H-%M"));
        let out_dir = Path::new(&dirname);

        Self {
            out_dir: out_dir.to_path_buf(),
        }
    }

    pub fn create_writer(&self, filename: &str) -> FileWriter {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let filename = Path::new(filename).file_name()
                    .expect("Failed to get file name");
                FileWriter {
                    writer: Cursor::new(Vec::new()),
                    path: filename.to_str().unwrap_or("aurora_file").to_string(),
                }
            } else {
                if !self.out_dir.exists() {
                    std::fs::create_dir_all(&self.out_dir)
                        .expect("Failed to create output directory");
                }

                let path = self.out_dir.clone().join(filename);
                let file = std::fs::File::create(&path)
                    .expect("Failed to create file");
                let writer = std::io::BufWriter::new(file);
                FileWriter {
                    writer: writer,
                }
            }
        }
    }

    pub async fn create_reader(&self, filename: &str) -> futures::io::Cursor<Vec<u8>> {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let location = web_sys::window().expect("should have a window").location();
                let url = format!(
                    "{}{}{}",
                    location.origin().expect("Failed to get origin"),
                    location.pathname().unwrap_or("/".to_string()),
                    filename
                );
                let byte_vec = reqwest::get(&url).await
                    .expect("Failed to fetch file")
                    .bytes().await
                    .expect("Failed to read bytes")
                    .to_vec();
                futures::io::Cursor::new(byte_vec)
            } else {
                let path = Path::new(filename);
                let file = std::fs::File::open(&path)
                    .expect(&format!("Failed to open file {path:?}"));
                let mut contents = Vec::new();
                let mut file_reader = std::io::BufReader::new(file);
                file_reader.read_to_end(&mut contents).expect("Failed to read file contents");
                futures::io::Cursor::new(contents)
            }
        }
    }
}
