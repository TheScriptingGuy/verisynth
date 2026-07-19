//! Rust PyO3 kernels for Verisynth deterministic keyed generation.
//!
//! Implements docs/ARCHITECTURE.md §1.1-1.2 exactly, bit-identical to
//! `verisynth/_reference.py` for `keyed_hash`/`keyed_uniforms`, and
//! agreeing with `inv_norm_cdf` within 1e-12 absolute.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod xmlstream;
use xmlstream::{count_xml_records, stream_xml_file, XmlBatchIter};

const GOLDEN: u64 = 0x9E3779B97F4A7C15;
const MIX_C1: u64 = 0xBF58476D1CE4E5B9;
const MIX_C2: u64 = 0x94D049BB133111EB;
const FNV_OFFSET: u64 = 0xCBF29CE484222325;
const FNV_PRIME: u64 = 0x100000001B3;

#[inline]
fn mix64(z0: u64) -> u64 {
    let mut z = z0;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_C1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_C2);
    z ^ (z >> 31)
}

#[inline]
fn fnv1a64(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h = (h ^ (*b as u64)).wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline]
fn cell_hash(seed: u64, ns_hash: u64, key: u64, draw: u64) -> u64 {
    let mut h = mix64(seed ^ GOLDEN);
    h = mix64(h ^ ns_hash);
    h = mix64(h ^ key);
    h = mix64(h ^ draw);
    h
}

#[inline]
fn uniform_from_hash(h: u64) -> f64 {
    ((h >> 11) as f64 + 0.5) * 2f64.powi(-53)
}

// --------------------------------------------------------------------------
// Acklam's rational approximation of the standard normal inverse CDF.
// Coefficients and Horner evaluation order per docs/ARCHITECTURE.md §1.2.
// --------------------------------------------------------------------------

const A: [f64; 6] = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
];
const B: [f64; 5] = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
];
const C: [f64; 6] = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
];
const D: [f64; 4] = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
];
const P_LOW: f64 = 0.02425;
const P_HIGH: f64 = 1.0 - P_LOW;

#[inline]
fn acklam_inv_norm_cdf(u: f64) -> f64 {
    if !(u > 0.0 && u < 1.0) {
        // Covers u <= 0, u >= 1, and NaN (all comparisons with NaN are
        // false, so `!(u > 0.0 && u < 1.0)` is true for NaN too).
        return f64::NAN;
    }

    if u < P_LOW {
        let q = (-2.0 * u.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if u <= P_HIGH {
        let q = u - 0.5;
        let r = q * q;
        let num = ((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5];
        let num = num * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -num / den
    }
}

#[pyfunction]
#[pyo3(signature = (seed, namespace, keys, draw))]
fn keyed_hash<'py>(
    py: Python<'py>,
    seed: u64,
    namespace: &str,
    keys: PyReadonlyArray1<'py, u64>,
    draw: u64,
) -> PyResult<Py<PyArray1<u64>>> {
    let ns_hash = fnv1a64(namespace);
    let keys_vec: Vec<u64> = keys.as_array().iter().copied().collect();

    let out = py.allow_threads(|| {
        keys_vec
            .into_iter()
            .map(|k| cell_hash(seed, ns_hash, k, draw))
            .collect::<Vec<u64>>()
    });

    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (seed, namespace, keys, draw))]
fn keyed_uniforms<'py>(
    py: Python<'py>,
    seed: u64,
    namespace: &str,
    keys: PyReadonlyArray1<'py, u64>,
    draw: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ns_hash = fnv1a64(namespace);
    let keys_vec: Vec<u64> = keys.as_array().iter().copied().collect();

    let out = py.allow_threads(|| {
        keys_vec
            .into_iter()
            .map(|k| uniform_from_hash(cell_hash(seed, ns_hash, k, draw)))
            .collect::<Vec<f64>>()
    });

    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn inv_norm_cdf<'py>(py: Python<'py>, u: PyReadonlyArray1<'py, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let u_vec: Vec<f64> = u.as_array().iter().copied().collect();

    let out = py.allow_threads(|| {
        u_vec
            .into_iter()
            .map(acklam_inv_norm_cdf)
            .collect::<Vec<f64>>()
    });

    Ok(out.into_pyarray(py).unbind())
}

#[pymodule]
fn verisynth_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keyed_hash, m)?)?;
    m.add_function(wrap_pyfunction!(keyed_uniforms, m)?)?;
    m.add_function(wrap_pyfunction!(inv_norm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(stream_xml_file, m)?)?;
    m.add_function(wrap_pyfunction!(count_xml_records, m)?)?;
    m.add_class::<XmlBatchIter>()?;
    Ok(())
}
