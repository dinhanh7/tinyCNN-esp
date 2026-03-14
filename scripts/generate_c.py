#!/usr/bin/env python3
"""
generate_c.py

Parse model_int8.onnx and generate pure-C inference engine files for ESP32 Arduino.

Output files (in esp32_inference/):
  - model_weights.h  : all weights, biases, scales, zero points
  - model.h          : structs, defines, function prototypes
  - model.c          : pure-C inference layer implementations
  - input_image.h    : real CIFAR-10 test image quantized to int8
  - main.ino         : Arduino sketch
"""
import os
import struct
import pickle
import numpy as np
import onnx
from onnx import numpy_helper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_MODEL = os.path.join(PROJECT_ROOT, "models", "model_int8.onnx")
OUT_DIR     = os.path.join(PROJECT_ROOT, "esp32_inference")

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_initializer(graph, name):
    """Return numpy array for a named initializer."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"Initializer '{name}' not found")

def arr_to_c_int8(name, arr, per_line=16):
    """Format int8 numpy array as a C static const array with PROGMEM."""
    flat = arr.flatten().astype(np.int8)
    lines = [f"static const int8_t {name}[{len(flat)}] PROGMEM = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        lines.append("    " + ", ".join(f"{v}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)

def arr_to_c_int32(name, arr, per_line=8):
    """Format int32 numpy array as a C static const array."""
    flat = arr.flatten().astype(np.int32)
    lines = [f"static const int32_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        lines.append("    " + ", ".join(f"{v}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)

def arr_to_c_float(name, arr, per_line=8):
    """Format float32 numpy array as a C static const array."""
    flat = arr.flatten().astype(np.float32)
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        lines.append("    " + ", ".join(f"{v:.10e}f" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)

def scalar_to_c_float(name, val):
    return f"static const float {name} = {float(val):.10e}f;"

def scalar_to_c_uint8(name, val):
    return f"static const uint8_t {name} = {int(val)};"

def scalar_to_c_int8(name, val):
    return f"static const int8_t {name} = {int(val)};"

# ---------------------------------------------------------------------------
# Layer descriptor: parsed from graph
# ---------------------------------------------------------------------------
class ConvLayer:
    def __init__(self, idx, name, w, b, w_scale, w_zp,
                 in_scale, in_zp, out_scale, out_zp,
                 in_ch, out_ch, kh, kw, stride, pad, group, spatial_in):
        self.idx = idx
        self.name = name
        self.w = w
        self.b = b
        self.w_scale = w_scale
        self.w_zp = w_zp
        self.in_scale = in_scale
        self.in_zp = in_zp
        self.out_scale = out_scale
        self.out_zp = out_zp
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kh = kh
        self.kw = kw
        self.stride = stride
        self.pad = pad
        self.group = group
        self.depthwise = (group == in_ch and group == out_ch and group > 1)
        self.spatial_in = spatial_in
        self.spatial_out = (spatial_in + 2 * pad - kh) // stride + 1

class GemmLayer:
    def __init__(self, name, w, b, w_scale, w_zp,
                 in_scale, in_zp, out_scale, out_zp,
                 in_features, out_features):
        self.name = name
        self.w = w
        self.b = b
        self.w_scale = w_scale
        self.w_zp = w_zp
        self.in_scale = in_scale
        self.in_zp = in_zp
        self.out_scale = out_scale
        self.out_zp = out_zp
        self.in_features = in_features
        self.out_features = out_features

class AddLayer:
    def __init__(self, name, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp, channels, spatial):
        self.name = name
        self.a_scale = a_scale
        self.a_zp = a_zp
        self.b_scale = b_scale
        self.b_zp = b_zp
        self.out_scale = out_scale
        self.out_zp = out_zp
        self.channels = channels
        self.spatial = spatial

# ---------------------------------------------------------------------------
# Parse ONNX graph
# ---------------------------------------------------------------------------
def parse_graph(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    g = lambda n: get_initializer(graph, n)

    # Track spatial sizes through the network
    # Input: 3x32x32
    spatial = 32

    convs = []
    add_layer = None
    gemm_layer = None

    # We need to track the output name -> spatial dim mapping
    spatial_map = {"input_quantized": 32}

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # input -> input_quantized  or  view -> view_quantized
            out_name = node.output[0]
            in_name = node.input[0]
            if in_name == "input":
                spatial_map[out_name] = 32
            # view_quantized is 1D, no spatial
            continue

        if node.op_type == "DequantizeLinear":
            continue

        if node.op_type == "ReduceMean":
            continue

        if node.op_type == "Reshape":
            continue

        if node.op_type == "QLinearConv":
            inp_names = list(node.input)
            # inp: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias
            x_name = inp_names[0]
            in_scale_v = float(g(inp_names[1]))
            in_zp_v = int(g(inp_names[2]))
            w = g(inp_names[3])
            w_scale = g(inp_names[4])
            w_zp = g(inp_names[5])
            out_scale_v = float(g(inp_names[6]))
            out_zp_v = int(g(inp_names[7]))
            b = g(inp_names[8])

            # Get attributes
            attrs = {}
            for attr in node.attribute:
                if attr.type == 2:  # INT
                    attrs[attr.name] = attr.i
                elif attr.type == 7:  # INTS
                    attrs[attr.name] = list(attr.ints)

            group = attrs.get("group", 1)
            strides = attrs.get("strides", [1, 1])
            pads = attrs.get("pads", [0, 0, 0, 0])
            stride = strides[0]
            pad = pads[0]

            out_ch, in_ch_per_group, kh, kw = w.shape
            in_ch = in_ch_per_group * group

            # Get input spatial from map
            sp_in = spatial_map.get(x_name, spatial)
            sp_out = (sp_in + 2 * pad - kh) // stride + 1

            idx = len(convs)
            layer = ConvLayer(
                idx=idx,
                name=node.name.replace("_quant", "").replace("node_", ""),
                w=w, b=b, w_scale=w_scale, w_zp=w_zp,
                in_scale=in_scale_v, in_zp=in_zp_v,
                out_scale=out_scale_v, out_zp=out_zp_v,
                in_ch=in_ch, out_ch=out_ch,
                kh=kh, kw=kw, stride=stride, pad=pad,
                group=group, spatial_in=sp_in,
            )
            convs.append(layer)

            # Update spatial map
            out_name = node.output[0]
            spatial_map[out_name] = sp_out
            spatial = sp_out

        elif node.op_type == "QLinearAdd":
            inp_names = list(node.input)
            a_scale = float(g(inp_names[1]))
            a_zp = int(g(inp_names[2]))
            b_scale = float(g(inp_names[4]))
            b_zp = int(g(inp_names[5]))
            out_scale = float(g(inp_names[6]))
            out_zp = int(g(inp_names[7]))

            # Figure out channels and spatial from the A input
            a_name = inp_names[0]
            sp = spatial_map.get(a_name, spatial)

            # After mbconv1 projection, channels = 16, spatial = 32
            # A = getitem_quantized (stem output), B = getitem_6_quantized (mbconv1 proj output)
            # Both are 16 channels
            add_layer = AddLayer(
                name="residual_add",
                a_scale=a_scale, a_zp=a_zp,
                b_scale=b_scale, b_zp=b_zp,
                out_scale=out_scale, out_zp=out_zp,
                channels=16, spatial=sp,
            )

            out_name = node.output[0]
            spatial_map[out_name] = sp

        elif node.op_type == "QGemm":
            inp_names = list(node.input)
            in_scale_v = float(g(inp_names[1]))
            in_zp_v = int(g(inp_names[2]))
            w = g(inp_names[3])
            w_scale = g(inp_names[4])
            w_zp = g(inp_names[5])
            b = g(inp_names[6])
            out_scale_v = float(g(inp_names[7]))
            out_zp_v = int(g(inp_names[8]))

            out_features, in_features = w.shape

            gemm_layer = GemmLayer(
                name="classifier",
                w=w, b=b, w_scale=w_scale, w_zp=w_zp,
                in_scale=in_scale_v, in_zp=in_zp_v,
                out_scale=out_scale_v, out_zp=out_zp_v,
                in_features=in_features, out_features=out_features,
            )

    return convs, add_layer, gemm_layer


# ---------------------------------------------------------------------------
# Generate model_weights.h
# ---------------------------------------------------------------------------
def generate_weights_h(convs, add_layer, gemm, path):
    lines = []
    lines.append("/* AUTO-GENERATED - DO NOT EDIT */")
    lines.append("#ifndef MODEL_WEIGHTS_H")
    lines.append("#define MODEL_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("/* Store large weight arrays in flash (PROGMEM) */")
    lines.append("#ifdef __AVR__")
    lines.append("  #include <avr/pgmspace.h>")
    lines.append("#else")
    lines.append("  #ifndef PROGMEM")
    lines.append("    #define PROGMEM")
    lines.append("  #endif")
    lines.append("#endif")
    lines.append("")

    # Input quantization
    lines.append("/* ---- Input quantization ---- */")
    lines.append(f"#define INPUT_SCALE  {convs[0].in_scale:.10e}f")
    lines.append(f"#define INPUT_ZP     {convs[0].in_zp}")
    lines.append("")

    # Conv layers
    for i, c in enumerate(convs):
        lines.append(f"/* ---- Conv layer {i}: {c.name} ---- */")
        lines.append(f"/* in={c.in_ch}x{c.spatial_in}x{c.spatial_in} out={c.out_ch}x{c.spatial_out}x{c.spatial_out}"
                      f" k={c.kh}x{c.kw} s={c.stride} p={c.pad} g={c.group}"
                      f" {'depthwise' if c.depthwise else 'standard'} */")

        # Weights
        lines.append(arr_to_c_int8(f"conv{i}_weight", c.w))
        lines.append(arr_to_c_int32(f"conv{i}_bias", c.b))
        lines.append(arr_to_c_float(f"conv{i}_w_scale", c.w_scale))
        lines.append(scalar_to_c_float(f"conv{i}_in_scale", c.in_scale))
        lines.append(scalar_to_c_uint8(f"conv{i}_in_zp", c.in_zp))
        lines.append(scalar_to_c_float(f"conv{i}_out_scale", c.out_scale))
        lines.append(scalar_to_c_uint8(f"conv{i}_out_zp", c.out_zp))
        lines.append("")

    # Add layer
    if add_layer:
        lines.append("/* ---- Residual Add ---- */")
        lines.append(scalar_to_c_float("add_a_scale", add_layer.a_scale))
        lines.append(scalar_to_c_uint8("add_a_zp", add_layer.a_zp))
        lines.append(scalar_to_c_float("add_b_scale", add_layer.b_scale))
        lines.append(scalar_to_c_uint8("add_b_zp", add_layer.b_zp))
        lines.append(scalar_to_c_float("add_out_scale", add_layer.out_scale))
        lines.append(scalar_to_c_uint8("add_out_zp", add_layer.out_zp))
        lines.append("")

    # Gemm layer
    if gemm:
        lines.append("/* ---- Dense (Gemm) layer ---- */")
        lines.append(arr_to_c_int8("dense_weight", gemm.w))
        lines.append(arr_to_c_int32("dense_bias", gemm.b))
        lines.append(arr_to_c_float("dense_w_scale", gemm.w_scale))
        lines.append(scalar_to_c_float("dense_in_scale", gemm.in_scale))
        lines.append(scalar_to_c_uint8("dense_in_zp", gemm.in_zp))
        lines.append(scalar_to_c_float("dense_out_scale", gemm.out_scale))
        lines.append(scalar_to_c_uint8("dense_out_zp", gemm.out_zp))
        lines.append("")

    lines.append("#endif /* MODEL_WEIGHTS_H */")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written {path} ({os.path.getsize(path)} bytes)")


# ---------------------------------------------------------------------------
# Generate model.h
# ---------------------------------------------------------------------------
def generate_model_h(convs, add_layer, gemm, path):
    # Calculate max buffer sizes
    # We need to know size of each intermediate tensor
    buf_sizes = []
    for c in convs:
        buf_sizes.append(c.in_ch * c.spatial_in * c.spatial_in)
        buf_sizes.append(c.out_ch * c.spatial_out * c.spatial_out)
    max_buf = max(buf_sizes)

    # Also need add_layer buffer for residual (same as stem output size)
    residual_size = 16 * 32 * 32  # stem output

    lines = []
    lines.append("/* AUTO-GENERATED - DO NOT EDIT */")
    lines.append("#ifndef MODEL_H")
    lines.append("#define MODEL_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("/* ---- Model dimensions ---- */")
    lines.append(f"#define INPUT_C     3")
    lines.append(f"#define INPUT_H     32")
    lines.append(f"#define INPUT_W     32")
    lines.append(f"#define INPUT_SIZE  (INPUT_C * INPUT_H * INPUT_W)   /* {3*32*32} */")
    lines.append(f"#define NUM_CLASSES 10")
    lines.append(f"#define NUM_CONV_LAYERS {len(convs)}")
    lines.append(f"#define MAX_BUF_SIZE {max_buf}")
    lines.append(f"#define RESIDUAL_SIZE {residual_size}")
    lines.append("")

    # Per-layer dimensions
    lines.append("/* ---- Per-layer dimensions ---- */")
    for i, c in enumerate(convs):
        prefix = f"L{i}"
        lines.append(f"#define {prefix}_IN_C   {c.in_ch}")
        lines.append(f"#define {prefix}_OUT_C  {c.out_ch}")
        lines.append(f"#define {prefix}_KH     {c.kh}")
        lines.append(f"#define {prefix}_KW     {c.kw}")
        lines.append(f"#define {prefix}_STRIDE {c.stride}")
        lines.append(f"#define {prefix}_PAD    {c.pad}")
        lines.append(f"#define {prefix}_GROUP  {c.group}")
        lines.append(f"#define {prefix}_SP_IN  {c.spatial_in}")
        lines.append(f"#define {prefix}_SP_OUT {c.spatial_out}")
        dw = 1 if c.depthwise else 0
        lines.append(f"#define {prefix}_DW     {dw}")
        lines.append("")

    # Dense layer
    if gemm:
        lines.append(f"#define DENSE_IN   {gemm.in_features}")
        lines.append(f"#define DENSE_OUT  {gemm.out_features}")
        lines.append("")

    # Global avg pool params (conv_head output)
    last_conv = convs[-1]
    lines.append(f"#define POOL_C   {last_conv.out_ch}")
    lines.append(f"#define POOL_H   {last_conv.spatial_out}")
    lines.append(f"#define POOL_W   {last_conv.spatial_out}")
    lines.append("")

    lines.append("/* ---- Function prototypes ---- */")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.append("void predict(const uint8_t *input_quant, float *output);")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append("#endif /* MODEL_H */")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written {path} ({os.path.getsize(path)} bytes)")

    return max_buf, residual_size


# ---------------------------------------------------------------------------
# Generate model.c
# ---------------------------------------------------------------------------
def generate_model_c(convs, add_layer, gemm, max_buf, residual_size, path):
    lines = []
    lines.append("/* AUTO-GENERATED - DO NOT EDIT */")
    lines.append("/*")
    lines.append(" * Pure-C INT8 inference engine for TinyMobileNet CIFAR-10")
    lines.append(" * Target: ESP32 (xtensa gcc, Arduino IDE)")
    lines.append(" * No malloc, no C++, only static arrays.")
    lines.append(" */")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("#include <string.h>")
    lines.append("#include <math.h>")
    lines.append('#include "model.h"')
    lines.append('#include "model_weights.h"')
    lines.append("")

    # Static buffers
    lines.append("/* ---- Static activation buffers (ping-pong) ---- */")
    lines.append(f"static uint8_t buf_a[MAX_BUF_SIZE];")
    lines.append(f"static uint8_t buf_b[MAX_BUF_SIZE];")
    lines.append(f"static uint8_t residual_buf[RESIDUAL_SIZE];")
    lines.append("")

    # Helper: clamp
    lines.append("/* ---- Helpers ---- */")
    lines.append("static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {")
    lines.append("    if (x < lo) return lo;")
    lines.append("    if (x > hi) return hi;")
    lines.append("    return x;")
    lines.append("}")
    lines.append("")

    # Read weight from PROGMEM
    lines.append("static inline int8_t read_weight(const int8_t *p, int idx) {")
    lines.append("#ifdef __AVR__")
    lines.append("    return (int8_t)pgm_read_byte(&p[idx]);")
    lines.append("#else")
    lines.append("    return p[idx];")
    lines.append("#endif")
    lines.append("}")
    lines.append("")

    # ----- QLinearConv (standard) -----
    lines.append("/* ---- Standard 2D Convolution (INT8, per-channel) ---- */")
    lines.append("static void conv2d_int8(")
    lines.append("    const uint8_t *input, uint8_t *output,")
    lines.append("    const int8_t *weight, const int32_t *bias, const float *w_scale,")
    lines.append("    float in_scale, uint8_t in_zp,")
    lines.append("    float out_scale, uint8_t out_zp,")
    lines.append("    int in_c, int out_c, int sp_in, int sp_out,")
    lines.append("    int kh, int kw, int stride, int pad")
    lines.append(") {")
    lines.append("    int oc, oh, ow, ic, fh, fw;")
    lines.append("    for (oc = 0; oc < out_c; oc++) {")
    lines.append("        float M = (in_scale * w_scale[oc]) / out_scale;")
    lines.append("        for (oh = 0; oh < sp_out; oh++) {")
    lines.append("            for (ow = 0; ow < sp_out; ow++) {")
    lines.append("                int32_t acc = bias[oc];")
    lines.append("                for (ic = 0; ic < in_c; ic++) {")
    lines.append("                    for (fh = 0; fh < kh; fh++) {")
    lines.append("                        for (fw = 0; fw < kw; fw++) {")
    lines.append("                            int ih = oh * stride - pad + fh;")
    lines.append("                            int iw = ow * stride - pad + fw;")
    lines.append("                            int32_t x_val = 0;")
    lines.append("                            if (ih >= 0 && ih < sp_in && iw >= 0 && iw < sp_in) {")
    lines.append("                                x_val = (int32_t)input[ic * sp_in * sp_in + ih * sp_in + iw] - (int32_t)in_zp;")
    lines.append("                            } else {")
    lines.append("                                x_val = -(int32_t)in_zp;")
    lines.append("                            }")
    lines.append("                            int w_idx = oc * (in_c * kh * kw) + ic * (kh * kw) + fh * kw + fw;")
    lines.append("                            int32_t w_val = (int32_t)read_weight(weight, w_idx);")
    lines.append("                            acc += x_val * w_val;")
    lines.append("                        }")
    lines.append("                    }")
    lines.append("                }")
    lines.append("                /* Requantize: out = clamp(round(acc * M) + out_zp, 0, 255) */")
    lines.append("                int32_t out_val = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;")
    lines.append("                output[oc * sp_out * sp_out + oh * sp_out + ow] = (uint8_t)clamp_i32(out_val, 0, 255);")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- Depthwise Conv2D -----
    lines.append("/* Depthwise 2D Convolution (INT8, per-channel) */")
    lines.append("static void depthwise_conv2d_int8(")
    lines.append("    const uint8_t *input, uint8_t *output,")
    lines.append("    const int8_t *weight, const int32_t *bias, const float *w_scale,")
    lines.append("    float in_scale, uint8_t in_zp,")
    lines.append("    float out_scale, uint8_t out_zp,")
    lines.append("    int channels, int sp_in, int sp_out,")
    lines.append("    int kh, int kw, int stride, int pad")
    lines.append(") {")
    lines.append("    int ch, oh, ow, fh, fw;")
    lines.append("    for (ch = 0; ch < channels; ch++) {")
    lines.append("        float M = (in_scale * w_scale[ch]) / out_scale;")
    lines.append("        for (oh = 0; oh < sp_out; oh++) {")
    lines.append("            for (ow = 0; ow < sp_out; ow++) {")
    lines.append("                int32_t acc = bias[ch];")
    lines.append("                for (fh = 0; fh < kh; fh++) {")
    lines.append("                    for (fw = 0; fw < kw; fw++) {")
    lines.append("                        int ih = oh * stride - pad + fh;")
    lines.append("                        int iw = ow * stride - pad + fw;")
    lines.append("                        int32_t x_val = 0;")
    lines.append("                        if (ih >= 0 && ih < sp_in && iw >= 0 && iw < sp_in) {")
    lines.append("                            x_val = (int32_t)input[ch * sp_in * sp_in + ih * sp_in + iw] - (int32_t)in_zp;")
    lines.append("                        } else {")
    lines.append("                            x_val = -(int32_t)in_zp;")
    lines.append("                        }")
    lines.append("                        int w_idx = ch * (kh * kw) + fh * kw + fw;")
    lines.append("                        int32_t w_val = (int32_t)read_weight(weight, w_idx);")
    lines.append("                        acc += x_val * w_val;")
    lines.append("                    }")
    lines.append("                }")
    lines.append("                int32_t out_val = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;")
    lines.append("                output[ch * sp_out * sp_out + oh * sp_out + ow] = (uint8_t)clamp_i32(out_val, 0, 255);")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- QLinearAdd -----
    lines.append("/* Quantized element-wise add */")
    lines.append("static void qadd_int8(")
    lines.append("    const uint8_t *a, const uint8_t *b, uint8_t *out,")
    lines.append("    float a_scale, uint8_t a_zp,")
    lines.append("    float b_scale, uint8_t b_zp,")
    lines.append("    float out_scale, uint8_t out_zp,")
    lines.append("    int size")
    lines.append(") {")
    lines.append("    float a_ratio = a_scale / out_scale;")
    lines.append("    float b_ratio = b_scale / out_scale;")
    lines.append("    int i;")
    lines.append("    for (i = 0; i < size; i++) {")
    lines.append("        float val = ((float)a[i] - (float)a_zp) * a_ratio")
    lines.append("                  + ((float)b[i] - (float)b_zp) * b_ratio")
    lines.append("                  + (float)out_zp;")
    lines.append("        int32_t ival = (int32_t)(val + 0.5f);")
    lines.append("        out[i] = (uint8_t)clamp_i32(ival, 0, 255);")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- Global Average Pool -----
    lines.append("/* Global average pool: uint8 -> float per channel */")
    lines.append("static void global_avg_pool_uint8(")
    lines.append("    const uint8_t *input, float *output,")
    lines.append("    float scale, uint8_t zp,")
    lines.append("    int channels, int h, int w")
    lines.append(") {")
    lines.append("    int c, y, x;")
    lines.append("    int hw = h * w;")
    lines.append("    for (c = 0; c < channels; c++) {")
    lines.append("        int32_t sum = 0;")
    lines.append("        for (y = 0; y < h; y++) {")
    lines.append("            for (x = 0; x < w; x++) {")
    lines.append("                sum += (int32_t)input[c * hw + y * w + x];")
    lines.append("            }")
    lines.append("        }")
    lines.append("        /* Dequantize the mean */")
    lines.append("        float mean_q = (float)sum / (float)hw;")
    lines.append("        output[c] = (mean_q - (float)zp) * scale;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- Dense (QGemm) -----
    lines.append("/* Dense layer: quantized -> float output */")
    lines.append("static void dense_int8(")
    lines.append("    const float *input_float,")
    lines.append("    float *output,")
    lines.append("    const int8_t *weight, const int32_t *bias, const float *w_scale,")
    lines.append("    float in_scale, uint8_t in_zp,")
    lines.append("    float out_scale, uint8_t out_zp,")
    lines.append("    int in_features, int out_features")
    lines.append(") {")
    lines.append("    int i, j;")
    lines.append("    /* Quantize float input to uint8 */")
    lines.append(f"    static uint8_t q_in[DENSE_IN];")
    lines.append("    for (i = 0; i < in_features; i++) {")
    lines.append("        float qf = input_float[i] / in_scale + (float)in_zp;")
    lines.append("        q_in[i] = (uint8_t)clamp_i32((int32_t)(qf + 0.5f), 0, 255);")
    lines.append("    }")
    lines.append("")
    lines.append("    /* Matrix multiply: weight is [out_features, in_features] (transB=1 in QGemm) */")
    lines.append("    for (i = 0; i < out_features; i++) {")
    lines.append("        int32_t acc = bias[i];")
    lines.append("        float M = (in_scale * w_scale[i]) / out_scale;")
    lines.append("        for (j = 0; j < in_features; j++) {")
    lines.append("            int32_t x_val = (int32_t)q_in[j] - (int32_t)in_zp;")
    lines.append("            int32_t w_val = (int32_t)read_weight(weight, i * in_features + j);")
    lines.append("            acc += x_val * w_val;")
    lines.append("        }")
    lines.append("        /* Dequantize output to float */")
    lines.append("        int32_t out_q = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;")
    lines.append("        output[i] = ((float)out_q - (float)out_zp) * out_scale;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- Softmax -----
    lines.append("/* Softmax */")
    lines.append("static void softmax(float *x, int n) {")
    lines.append("    int i;")
    lines.append("    float max_val = x[0];")
    lines.append("    for (i = 1; i < n; i++) {")
    lines.append("        if (x[i] > max_val) max_val = x[i];")
    lines.append("    }")
    lines.append("    float sum = 0.0f;")
    lines.append("    for (i = 0; i < n; i++) {")
    lines.append("        x[i] = expf(x[i] - max_val);")
    lines.append("        sum += x[i];")
    lines.append("    }")
    lines.append("    for (i = 0; i < n; i++) {")
    lines.append("        x[i] /= sum;")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ----- Main predict() -----
    lines.append("/* Main inference function */")
    lines.append("void predict(const uint8_t *input_quant, float *output) {")
    lines.append("    uint8_t *cur_in  = buf_a;")
    lines.append("    uint8_t *cur_out = buf_b;")
    lines.append("    uint8_t *tmp;")
    lines.append("")

    # Copy input to buf_a
    lines.append("    /* Copy quantized input to working buffer */")
    lines.append("    memcpy(cur_in, input_quant, INPUT_SIZE);")
    lines.append("")

    # Now emit each layer call
    # The graph is:
    # conv0 (stem)  -> save to residual_buf for skip
    # conv1 (mbconv1 depthwise)
    # conv2 (mbconv1 pointwise)
    # qadd (residual_buf + conv2 output)
    # conv3..conv13 (remaining convs)
    # global_avg_pool
    # dense

    # conv0: stem
    c = convs[0]
    lines.append(f"    /* Layer 0: Stem Conv {c.in_ch}->{c.out_ch} {c.kh}x{c.kw} s={c.stride} p={c.pad} */")
    lines.append(f"    conv2d_int8(cur_in, cur_out,")
    lines.append(f"        conv0_weight, conv0_bias, conv0_w_scale,")
    lines.append(f"        conv0_in_scale, conv0_in_zp, conv0_out_scale, conv0_out_zp,")
    lines.append(f"        L0_IN_C, L0_OUT_C, L0_SP_IN, L0_SP_OUT,")
    lines.append(f"        L0_KH, L0_KW, L0_STRIDE, L0_PAD);")
    lines.append(f"    tmp = cur_in; cur_in = cur_out; cur_out = tmp;")
    lines.append("")

    # Save stem output for residual
    lines.append(f"    /* Save stem output for skip connection */")
    lines.append(f"    memcpy(residual_buf, cur_in, L0_OUT_C * L0_SP_OUT * L0_SP_OUT);")
    lines.append("")

    # conv1: mbconv1 depthwise
    c = convs[1]
    lines.append(f"    /* Layer 1: MBConv1 Depthwise {c.in_ch}ch {c.kh}x{c.kw} s={c.stride} p={c.pad} */")
    lines.append(f"    depthwise_conv2d_int8(cur_in, cur_out,")
    lines.append(f"        conv1_weight, conv1_bias, conv1_w_scale,")
    lines.append(f"        conv1_in_scale, conv1_in_zp, conv1_out_scale, conv1_out_zp,")
    lines.append(f"        L1_IN_C, L1_SP_IN, L1_SP_OUT,")
    lines.append(f"        L1_KH, L1_KW, L1_STRIDE, L1_PAD);")
    lines.append(f"    tmp = cur_in; cur_in = cur_out; cur_out = tmp;")
    lines.append("")

    # conv2: mbconv1 pointwise
    c = convs[2]
    lines.append(f"    /* Layer 2: MBConv1 Pointwise {c.in_ch}->{c.out_ch} 1x1 */")
    lines.append(f"    conv2d_int8(cur_in, cur_out,")
    lines.append(f"        conv2_weight, conv2_bias, conv2_w_scale,")
    lines.append(f"        conv2_in_scale, conv2_in_zp, conv2_out_scale, conv2_out_zp,")
    lines.append(f"        L2_IN_C, L2_OUT_C, L2_SP_IN, L2_SP_OUT,")
    lines.append(f"        L2_KH, L2_KW, L2_STRIDE, L2_PAD);")
    lines.append(f"    tmp = cur_in; cur_in = cur_out; cur_out = tmp;")
    lines.append("")

    # Residual add
    lines.append(f"    /* Residual add: stem_output + mbconv1_output */")
    lines.append(f"    qadd_int8(residual_buf, cur_in, cur_out,")
    lines.append(f"        add_a_scale, add_a_zp,")
    lines.append(f"        add_b_scale, add_b_zp,")
    lines.append(f"        add_out_scale, add_out_zp,")
    lines.append(f"        {add_layer.channels} * {add_layer.spatial} * {add_layer.spatial});")
    lines.append(f"    tmp = cur_in; cur_in = cur_out; cur_out = tmp;")
    lines.append("")

    # Remaining conv layers (3..13)
    for i in range(3, len(convs)):
        c = convs[i]
        dw_str = "Depthwise" if c.depthwise else "Conv"
        layer_name = c.name
        lines.append(f"    /* Layer {i}: {dw_str} {c.in_ch}->{c.out_ch} {c.kh}x{c.kw} s={c.stride} p={c.pad} */")

        if c.depthwise:
            lines.append(f"    depthwise_conv2d_int8(cur_in, cur_out,")
            lines.append(f"        conv{i}_weight, conv{i}_bias, conv{i}_w_scale,")
            lines.append(f"        conv{i}_in_scale, conv{i}_in_zp, conv{i}_out_scale, conv{i}_out_zp,")
            lines.append(f"        L{i}_IN_C, L{i}_SP_IN, L{i}_SP_OUT,")
            lines.append(f"        L{i}_KH, L{i}_KW, L{i}_STRIDE, L{i}_PAD);")
        else:
            lines.append(f"    conv2d_int8(cur_in, cur_out,")
            lines.append(f"        conv{i}_weight, conv{i}_bias, conv{i}_w_scale,")
            lines.append(f"        conv{i}_in_scale, conv{i}_in_zp, conv{i}_out_scale, conv{i}_out_zp,")
            lines.append(f"        L{i}_IN_C, L{i}_OUT_C, L{i}_SP_IN, L{i}_SP_OUT,")
            lines.append(f"        L{i}_KH, L{i}_KW, L{i}_STRIDE, L{i}_PAD);")

        lines.append(f"    tmp = cur_in; cur_in = cur_out; cur_out = tmp;")
        lines.append("")

    # Global average pool
    last_conv = convs[-1]
    lines.append(f"    /* Global Average Pool: {last_conv.out_ch}x{last_conv.spatial_out}x{last_conv.spatial_out} -> {last_conv.out_ch} */")
    lines.append(f"    static float pool_out[POOL_C];")
    lines.append(f"    global_avg_pool_uint8(cur_in, pool_out,")
    lines.append(f"        conv{len(convs)-1}_out_scale, conv{len(convs)-1}_out_zp,")
    lines.append(f"        POOL_C, POOL_H, POOL_W);")
    lines.append("")

    # Dense
    lines.append(f"    /* Dense: {gemm.in_features} -> {gemm.out_features} */")
    lines.append(f"    dense_int8(pool_out, output,")
    lines.append(f"        dense_weight, dense_bias, dense_w_scale,")
    lines.append(f"        dense_in_scale, dense_in_zp,")
    lines.append(f"        dense_out_scale, dense_out_zp,")
    lines.append(f"        DENSE_IN, DENSE_OUT);")
    lines.append("")

    # Softmax
    lines.append(f"    /* Softmax */")
    lines.append(f"    softmax(output, NUM_CLASSES);")
    lines.append("}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written {path} ({os.path.getsize(path)} bytes)")


# ---------------------------------------------------------------------------
# Generate input_image.h
# ---------------------------------------------------------------------------
def generate_input_image_h(convs, path):
    """Load a real CIFAR-10 test image, quantize, and save as C array."""
    # Try to load real CIFAR-10 test set
    cifar_dir = os.path.join(PROJECT_ROOT, "data", "cifar-10-batches-py")
    test_batch = os.path.join(cifar_dir, "test_batch")

    if os.path.exists(test_batch):
        with open(test_batch, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]  # (10000, 3072)
        labels = batch[b"labels"]

        # Pick image index 0
        idx = 0
        img = images[idx].reshape(3, 32, 32).astype(np.float32) / 255.0
        label = labels[idx]

        # Normalize (same as train.py)
        mean = CIFAR10_MEAN.reshape(3, 1, 1)
        std = CIFAR10_STD.reshape(3, 1, 1)
        img_norm = (img - mean) / std

        # Quantize to uint8 using input scale and zero_point
        in_scale = convs[0].in_scale
        in_zp = convs[0].in_zp
        img_q = np.clip(np.round(img_norm / in_scale + in_zp), 0, 255).astype(np.uint8)
        img_flat = img_q.flatten()

        class_name = CIFAR10_CLASSES[label]
        print(f"  Test image: index={idx}, label={label} ({class_name})")
    else:
        # Fallback: random
        print(f"  WARNING: CIFAR-10 test set not found, using random image")
        img_flat = np.random.randint(0, 256, 3072).astype(np.uint8)
        label = -1
        class_name = "unknown"

    lines = []
    lines.append("/* Generated by generate_c.py */")
    lines.append("#ifndef INPUT_IMAGE_H")
    lines.append("#define INPUT_IMAGE_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append('#include "model.h"')
    lines.append("")
    lines.append(f"/* CIFAR-10 test image, class: {label} ({class_name}) */")
    lines.append(f"/* Quantized with scale={convs[0].in_scale:.10e}, zp={convs[0].in_zp} */")
    lines.append(f"#define EXPECTED_CLASS {label}")
    lines.append(f'#define EXPECTED_CLASS_NAME "{class_name}"')
    lines.append("")

    # Write as uint8 array
    lines.append(f"static const uint8_t input_image[INPUT_SIZE] = {{")
    per_line = 16
    for i in range(0, len(img_flat), per_line):
        chunk = img_flat[i:i+per_line]
        lines.append("    " + ", ".join(f"{v}" for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* INPUT_IMAGE_H */")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written {path} ({os.path.getsize(path)} bytes)")

    return img_flat, label, class_name


# ---------------------------------------------------------------------------
# Generate main.ino
# ---------------------------------------------------------------------------
def generate_main_ino(path):
    code = '''/* AUTO-GENERATED - DO NOT EDIT */
/*
 * main.ino - Arduino sketch for TinyMobileNet CIFAR-10 inference on ESP32
 *
 * Runs a single inference on a test image and prints results via Serial.
 */

extern "C" {
#include "model.h"
#include "model.c"
}
#include "input_image.h"

static float output[NUM_CLASSES];

/* CIFAR-10 class names */
static const char *class_names[NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Serial.println("========================================");
    Serial.println("TinyMobileNet CIFAR-10 INT8 Inference");
    Serial.println("========================================");
    Serial.print("Expected class: ");
    Serial.print(EXPECTED_CLASS);
    Serial.print(" (");
    Serial.print(EXPECTED_CLASS_NAME);
    Serial.println(")");
    Serial.println();

    /* Run inference */
    unsigned long t0 = millis();
    predict(input_image, output);
    unsigned long t1 = millis();

    Serial.print("Inference time: ");
    Serial.print(t1 - t0);
    Serial.println(" ms");
    Serial.println();

    /* Print class probabilities */
    Serial.println("Class probabilities:");
    int best_class = 0;
    float best_prob = output[0];
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print("  [");
        Serial.print(i);
        Serial.print("] ");
        Serial.print(class_names[i]);
        Serial.print(": ");
        Serial.print(output[i] * 100.0f, 2);
        Serial.println("%");
        if (output[i] > best_prob) {
            best_prob = output[i];
            best_class = i;
        }
    }

    Serial.println();
    Serial.print(">> Predicted class: ");
    Serial.print(best_class);
    Serial.print(" (");
    Serial.print(class_names[best_class]);
    Serial.print(") with ");
    Serial.print(best_prob * 100.0f, 2);
    Serial.println("% confidence");

    if (best_class == EXPECTED_CLASS) {
        Serial.println(">> CORRECT!");
    } else {
        Serial.println(">> MISMATCH with expected class");
    }

    Serial.println("========================================");
}

void loop() {
    /* Nothing to do - single inference only */
    delay(10000);
}
'''
    with open(path, "w") as f:
        f.write(code)
    print(f"  Written {path} ({os.path.getsize(path)} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("C Code Generator for INT8 ONNX Model")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    INCLUDE_DIR = os.path.join(OUT_DIR, "include")
    SRC_DIR = os.path.join(OUT_DIR, "src")
    os.makedirs(INCLUDE_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)

    print("Parsing ONNX model...")
    convs, add_layer, gemm = parse_graph(ONNX_MODEL)
    print(f"  Found {len(convs)} conv layers, {'1 add' if add_layer else 'no add'}, {'1 gemm' if gemm else 'no gemm'}")

    # Print layer summary
    for i, c in enumerate(convs):
        dw = "DW" if c.depthwise else "  "
        print(f"  conv{i}: {dw} {c.in_ch:3d}->{c.out_ch:3d} {c.kh}x{c.kw} s={c.stride} p={c.pad} "
              f"spatial {c.spatial_in}x{c.spatial_in}->{c.spatial_out}x{c.spatial_out}")

    print("Generating model_weights.h...")
    generate_weights_h(convs, add_layer, gemm,
                       os.path.join(INCLUDE_DIR, "model_weights.h"))

    print("Generating model.h...")
    max_buf, res_size = generate_model_h(convs, add_layer, gemm,
                                          os.path.join(INCLUDE_DIR, "model.h"))
    print(f"  Max buffer size: {max_buf} bytes")
    print(f"  Residual buffer: {res_size} bytes")
    print(f"  Total static RAM (buffers): {max_buf * 2 + res_size} bytes")

    print("Generating model.c...")
    generate_model_c(convs, add_layer, gemm, max_buf, res_size,
                      os.path.join(SRC_DIR, "model.c"))

    print("Generating input_image.h...")
    img, label, class_name = generate_input_image_h(
        convs, os.path.join(INCLUDE_DIR, "input_image.h"))

    print("Code generation complete.")
    print(f"  Output: {INCLUDE_DIR}/ (headers) + {SRC_DIR}/ (source)")

    total_weight = sum(c.w.size + c.b.size for c in convs)
    if gemm:
        total_weight += gemm.w.size + gemm.b.size
    print(f"  Total weight parameters: {total_weight:,d}")
    print(f"  Test image class: {label} ({class_name})")
    print("=" * 60)


if __name__ == "__main__":
    main()
