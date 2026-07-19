//! Streaming XML record reading for `verisynth.xmlstream`.
//!
//! Implements the record-flattening semantics documented as normative in
//! `verisynth/xmlstream.py` (`_iter_records_reference` / `flatten_xml_record`)
//! bit-for-bit: a record is each direct child element of the document root;
//! nested containers are flattened depth-first with collision fallback keys
//! of the form `{owner_tag}_{leaf_tag}`; tag names are reduced to their local
//! name (text after the last `:`); attributes and mixed container text are
//! ignored.
//!
//! Parsing is done with a bounded-memory `quick_xml::Reader` over a buffered
//! file reader: only the current record's subtree and the current batch's
//! columns are held in memory, never the whole document. The GIL is released
//! for the whole parse loop that fills one batch (or performs a full count),
//! and re-acquired only to build the returned Python lists.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::BufReader;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use quick_xml::events::Event;
use quick_xml::Reader;

// ----------------------------------------------------------------------
// Error plumbing: both I/O and quick-xml parse errors surface as a Python
// ValueError carrying the file path and the underlying error message.
// ----------------------------------------------------------------------

#[derive(Debug)]
enum XmlStreamError {
    Io(std::io::Error),
    Xml(quick_xml::Error),
}

impl fmt::Display for XmlStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XmlStreamError::Io(e) => write!(f, "{e}"),
            XmlStreamError::Xml(e) => write!(f, "{e}"),
        }
    }
}

impl From<std::io::Error> for XmlStreamError {
    fn from(e: std::io::Error) -> Self {
        XmlStreamError::Io(e)
    }
}

impl From<quick_xml::Error> for XmlStreamError {
    fn from(e: quick_xml::Error) -> Self {
        XmlStreamError::Xml(e)
    }
}

impl From<quick_xml::encoding::EncodingError> for XmlStreamError {
    fn from(e: quick_xml::encoding::EncodingError) -> Self {
        XmlStreamError::Xml(quick_xml::Error::from(e))
    }
}

fn parse_error(path: &str, e: XmlStreamError) -> PyErr {
    PyValueError::new_err(format!("failed to parse XML file '{path}': {e}"))
}

// ----------------------------------------------------------------------
// In-memory record subtree + the exact `flatten_xml_record` port.
// ----------------------------------------------------------------------

/// Local name: raw quick-xml tag names are always `prefix:tag` (no `{uri}`
/// expansion), so the local name is everything after the last `:`.
fn local_name(raw: &[u8]) -> String {
    let s = String::from_utf8_lossy(raw);
    match s.rfind(':') {
        Some(i) => s[i + 1..].to_string(),
        None => s.into_owned(),
    }
}

struct XmlNode {
    tag: String,
    text: String,
    children: Vec<XmlNode>,
}

impl XmlNode {
    fn new(tag: String) -> Self {
        XmlNode {
            tag,
            text: String::new(),
            children: Vec::new(),
        }
    }
}

/// Port of `flatten_xml_record`: flattens `elem`'s direct+nested children into
/// an ordered (first-seen) list of (key, value) pairs. Assignment to an
/// existing key overwrites the value in place without moving its position,
/// matching Python dict-assignment semantics exactly (this matters when a
/// fallback/prefixed key itself collides with one already present).
fn flatten_record(elem: &XmlNode) -> (Vec<String>, Vec<Option<String>>) {
    let mut names: Vec<String> = Vec::new();
    let mut values: Vec<Option<String>> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    fn assign(
        key: String,
        value: Option<String>,
        names: &mut Vec<String>,
        values: &mut Vec<Option<String>>,
        index: &mut HashMap<String, usize>,
    ) {
        if let Some(&idx) = index.get(&key) {
            values[idx] = value;
        } else {
            index.insert(key.clone(), names.len());
            names.push(key);
            values.push(value);
        }
    }

    for child in &elem.children {
        if !child.children.is_empty() {
            let (sub_names, sub_values) = flatten_record(child);
            for (k, v) in sub_names.into_iter().zip(sub_values) {
                let key = if index.contains_key(&k) {
                    format!("{}_{}", child.tag, k)
                } else {
                    k
                };
                assign(key, v, &mut names, &mut values, &mut index);
            }
        } else {
            let trimmed = child.text.trim();
            let value = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
            let key = if index.contains_key(&child.tag) {
                format!("{}_{}", elem.tag, child.tag)
            } else {
                child.tag.clone()
            };
            assign(key, value, &mut names, &mut values, &mut index);
        }
    }
    (names, values)
}

// ----------------------------------------------------------------------
// Streaming batch reader: pure-Rust core (no Python types) so it can run
// entirely inside `Python::allow_threads`.
// ----------------------------------------------------------------------

struct XmlBatchInner {
    reader: Reader<BufReader<File>>,
    buf: Vec<u8>,
    depth: u64,
    stack: Vec<XmlNode>,
    batch_limit: usize,
    batch_rows: usize,
    batch_names: Vec<String>,
    batch_index: HashMap<String, usize>,
    batch_columns: Vec<Vec<Option<String>>>,
}

impl XmlBatchInner {
    fn open(path: &str, batch_limit: usize) -> Result<Self, XmlStreamError> {
        let file = File::open(path)?;
        let reader = Reader::from_reader(BufReader::new(file));
        Ok(XmlBatchInner {
            reader,
            buf: Vec::with_capacity(4096),
            depth: 0,
            stack: Vec::new(),
            batch_limit,
            batch_rows: 0,
            batch_names: Vec::new(),
            batch_index: HashMap::new(),
            batch_columns: Vec::new(),
        })
    }

    /// Add one flattened record's (names, values) into the batch accumulator,
    /// extending the ordered column schema (first-seen order across the
    /// batch) and back-filling `None` for rows/columns that don't have it.
    fn push_record(&mut self, names: Vec<String>, values: Vec<Option<String>>) {
        for name in &names {
            if !self.batch_index.contains_key(name) {
                let idx = self.batch_names.len();
                self.batch_index.insert(name.clone(), idx);
                self.batch_names.push(name.clone());
                self.batch_columns.push(vec![None; self.batch_rows]);
            }
        }
        let mut rec_map: HashMap<String, Option<String>> =
            names.into_iter().zip(values).collect();
        for (i, col_name) in self.batch_names.iter().enumerate() {
            let val = rec_map.remove(col_name).unwrap_or(None);
            self.batch_columns[i].push(val);
        }
        self.batch_rows += 1;
    }

    /// Parse events until the current batch reaches `batch_limit` records or
    /// the file ends. Returns `true` if the (possibly partial, final) batch
    /// has at least one row, `false` at a clean end-of-stream with nothing
    /// left to yield.
    fn fill_batch(&mut self) -> Result<bool, XmlStreamError> {
        self.batch_rows = 0;
        self.batch_names.clear();
        self.batch_index.clear();
        self.batch_columns.clear();

        loop {
            let event = self.reader.read_event_into(&mut self.buf)?;
            let mut record_ready = false;
            match event {
                Event::Start(e) => {
                    self.depth += 1;
                    if self.depth >= 2 {
                        self.stack.push(XmlNode::new(local_name(e.name().as_ref())));
                    }
                }
                Event::Empty(e) => {
                    // Self-closing elements never change `depth` (no paired
                    // End event); their nesting level is `depth + 1`.
                    let virtual_depth = self.depth + 1;
                    if virtual_depth == 2 {
                        let node = XmlNode::new(local_name(e.name().as_ref()));
                        let (names, values) = flatten_record(&node);
                        self.push_record(names, values);
                        record_ready = true;
                    } else if virtual_depth > 2 {
                        let node = XmlNode::new(local_name(e.name().as_ref()));
                        if let Some(parent) = self.stack.last_mut() {
                            parent.children.push(node);
                        }
                    }
                    // virtual_depth == 1: the root itself is self-closed
                    // (e.g. `<root/>`) -> zero records, nothing to do.
                }
                Event::Text(e) => {
                    if let Some(top) = self.stack.last_mut() {
                        let text = e.unescape()?;
                        top.text.push_str(&text);
                    }
                }
                Event::CData(e) => {
                    if let Some(top) = self.stack.last_mut() {
                        let text = e.decode()?;
                        top.text.push_str(&text);
                    }
                }
                Event::End(_) => {
                    self.depth -= 1;
                    if self.depth == 1 {
                        if let Some(node) = self.stack.pop() {
                            let (names, values) = flatten_record(&node);
                            self.push_record(names, values);
                            record_ready = true;
                        }
                    } else if self.depth > 1 {
                        if let Some(node) = self.stack.pop() {
                            if let Some(parent) = self.stack.last_mut() {
                                parent.children.push(node);
                            }
                        }
                    }
                    // depth == 0: root closed.
                }
                Event::Eof => {
                    self.buf.clear();
                    return Ok(self.batch_rows > 0);
                }
                _ => {}
            }
            self.buf.clear();
            if record_ready && self.batch_rows >= self.batch_limit {
                return Ok(true);
            }
        }
    }
}

fn count_records_inner(path: &str) -> Result<u64, XmlStreamError> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(BufReader::new(file));
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut depth: u64 = 0;
    let mut count: u64 = 0;
    loop {
        match reader.read_event_into(&mut buf)? {
            Event::Start(_) => depth += 1,
            Event::Empty(_) => {
                if depth == 1 {
                    count += 1;
                }
            }
            Event::End(_) => {
                depth -= 1;
                if depth == 1 {
                    count += 1;
                }
            }
            Event::Eof => {
                buf.clear();
                return Ok(count);
            }
            _ => {}
        }
        buf.clear();
    }
}

// ----------------------------------------------------------------------
// PyO3 surface.
// ----------------------------------------------------------------------

#[pyclass]
pub struct XmlBatchIter {
    inner: XmlBatchInner,
    path: String,
    finished: bool,
}

#[pymethods]
impl XmlBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Option<(Py<PyList>, Py<PyList>)>> {
        if slf.finished {
            return Ok(None);
        }
        let path = slf.path.clone();
        // SAFETY: `inner_ptr` only outlives this call, which blocks (holding
        // `slf` on the stack) until the closure returns; no other code can
        // observe or mutate `slf.inner` while the GIL is released here. The
        // wrapper only exists to satisfy `allow_threads`'s `Send` bound for
        // what is, in practice, a single-threaded borrow handoff.
        struct SendPtr(*mut XmlBatchInner);
        unsafe impl Send for SendPtr {}
        let inner_ptr = SendPtr(&mut slf.inner);
        let result = py.allow_threads(move || {
            let inner_ptr = inner_ptr; // force whole-struct capture (2021 disjoint capture)
            unsafe { (*inner_ptr.0).fill_batch() }
        });
        match result {
            Err(e) => Err(parse_error(&path, e)),
            Ok(false) => {
                slf.finished = true;
                Ok(None)
            }
            Ok(true) => {
                let names = std::mem::take(&mut slf.inner.batch_names);
                let columns = std::mem::take(&mut slf.inner.batch_columns);
                let py_names = PyList::new(py, names)?;
                let py_columns = PyList::empty(py);
                for col in columns {
                    py_columns.append(PyList::new(py, col)?)?;
                }
                Ok(Some((py_names.unbind(), py_columns.unbind())))
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (path, batch_rows))]
pub fn stream_xml_file(path: String, batch_rows: usize) -> PyResult<XmlBatchIter> {
    let inner = XmlBatchInner::open(&path, batch_rows).map_err(|e| parse_error(&path, e))?;
    Ok(XmlBatchIter {
        inner,
        path,
        finished: false,
    })
}

#[pyfunction]
#[pyo3(signature = (path))]
pub fn count_xml_records(py: Python<'_>, path: String) -> PyResult<u64> {
    let path_for_result = path.clone();
    py.allow_threads(move || count_records_inner(&path))
        .map_err(|e| parse_error(&path_for_result, e))
}
