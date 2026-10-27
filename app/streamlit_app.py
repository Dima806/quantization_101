"""Streamlit app: four interactive views of quantization.

1. Quantize-a-matrix  2. Outlier breaker  3. Bit-width trade-off  4. Scheme comparison.
Run with ``make run``.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.bits.widths import size_ratio
from src.evaluation.comparison import arena
from src.evaluation.reconstruction import reconstruction_error
from src.models.weights import planted_outlier_matrix, smooth_weight_matrix
from src.quant.dequant import dequantize
from src.quant.granularity import quantize
from src.visualisation import (
    plot_error_vs_bits,
    plot_quantization_buckets,
    plot_weight_distribution,
)

st.set_page_config(page_title="Quantization 101", layout="wide")
st.title("Quantization 101")
st.caption(
    "Throw away three quarters of every number and barely notice — until an outlier shows up."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Quantize a matrix", "Outlier breaker", "Bit-width trade-off", "Scheme comparison"]
)


with tab1:
    st.header("Quantize a matrix")
    col = st.columns(3)
    bits = col[0].selectbox("Bit-width", [8, 4], key="t1_bits")
    scheme = col[1].selectbox("Scheme", ["absmax", "zeropoint"], key="t1_scheme")
    granularity = col[2].selectbox("Granularity", ["per_tensor", "per_channel"], key="t1_gran")

    matrix = smooth_weight_matrix(128, 128, seed=0)
    qa = quantize(matrix, bits=int(bits), scheme=scheme, granularity=granularity, axis=1)
    err = reconstruction_error(matrix, dequantize(qa))

    left, right = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plot_weight_distribution(matrix, ax=ax1)
    left.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_quantization_buckets(matrix, qa, ax=ax2)
    right.pyplot(fig2)

    m1, m2, m3 = st.columns(3)
    m1.metric("Relative error", f"{err['rel_fro']:.4%}")
    m2.metric("SNR (dB)", f"{err['snr_db']:.1f}")
    m3.metric("Size vs fp32", f"{size_ratio(int(bits)):.3f}x")


with tab2:
    st.header("Outlier breaker")
    scale = st.slider("Outlier size (x ordinary max)", 1.0, 50.0, 20.0, key="t2_scale")
    matrix, mask = planted_outlier_matrix(128, 128, outlier_scale=float(scale), seed=0)

    results = {}
    for gran in ("per_tensor", "per_channel"):
        qa = quantize(matrix, bits=8, scheme="absmax", granularity=gran, axis=1)
        ordinary_err = np.abs(matrix - dequantize(qa))[~mask]
        results[gran] = float(np.mean(ordinary_err))

    c1, c2 = st.columns(2)
    c1.metric("Per-tensor error (ordinary weights)", f"{results['per_tensor']:.2e}")
    c2.metric("Per-channel error (ordinary weights)", f"{results['per_channel']:.2e}")
    st.write(
        "One outlier stretches the per-tensor scale and collapses the ordinary weights. "
        "Per-channel scales confine the damage to a single row."
    )


with tab3:
    st.header("Bit-width trade-off")
    matrix = smooth_weight_matrix(128, 128, seed=0)
    table = arena(matrix)
    st.dataframe(table, use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_error_vs_bits(table, ax=ax)
    st.pyplot(fig)


with tab4:
    st.header("Scheme comparison")
    use_outlier = st.checkbox("Inject an outlier", value=True, key="t4_outlier")
    if use_outlier:
        matrix, _ = planted_outlier_matrix(128, 128, outlier_scale=20.0, seed=0)
    else:
        matrix = smooth_weight_matrix(128, 128, seed=0)
    st.dataframe(arena(matrix), use_container_width=True)
