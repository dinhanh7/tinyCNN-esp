/* AUTO-GENERATED - DO NOT EDIT */
/*
 * Pure-C INT8 inference engine for TinyMobileNet CIFAR-10
 * Target: ESP32 (xtensa gcc, Arduino IDE)
 * No malloc, no C++, only static arrays.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "model_weights.h"

/* ---- Static activation buffers (ping-pong) ---- */
static uint8_t buf_a[MAX_BUF_SIZE];
static uint8_t buf_b[MAX_BUF_SIZE];
static uint8_t residual_buf[RESIDUAL_SIZE];

/* ---- Helpers ---- */
static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline int8_t read_weight(const int8_t *p, int idx) {
#ifdef __AVR__
    return (int8_t)pgm_read_byte(&p[idx]);
#else
    return p[idx];
#endif
}

/* ---- Standard 2D Convolution (INT8, per-channel) ---- */
static void conv2d_int8(
    const uint8_t *input, uint8_t *output,
    const int8_t *weight, const int32_t *bias, const float *w_scale,
    float in_scale, uint8_t in_zp,
    float out_scale, uint8_t out_zp,
    int in_c, int out_c, int sp_in, int sp_out,
    int kh, int kw, int stride, int pad
) {
    int oc, oh, ow, ic, fh, fw;
    for (oc = 0; oc < out_c; oc++) {
        float M = (in_scale * w_scale[oc]) / out_scale;
        for (oh = 0; oh < sp_out; oh++) {
            for (ow = 0; ow < sp_out; ow++) {
                int32_t acc = bias[oc];
                for (ic = 0; ic < in_c; ic++) {
                    for (fh = 0; fh < kh; fh++) {
                        for (fw = 0; fw < kw; fw++) {
                            int ih = oh * stride - pad + fh;
                            int iw = ow * stride - pad + fw;
                            int32_t x_val = 0;
                            if (ih >= 0 && ih < sp_in && iw >= 0 && iw < sp_in) {
                                x_val = (int32_t)input[ic * sp_in * sp_in + ih * sp_in + iw] - (int32_t)in_zp;
                            } else {
                                x_val = -(int32_t)in_zp;
                            }
                            int w_idx = oc * (in_c * kh * kw) + ic * (kh * kw) + fh * kw + fw;
                            int32_t w_val = (int32_t)read_weight(weight, w_idx);
                            acc += x_val * w_val;
                        }
                    }
                }
                /* Requantize: out = clamp(round(acc * M) + out_zp, 0, 255) */
                int32_t out_val = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;
                output[oc * sp_out * sp_out + oh * sp_out + ow] = (uint8_t)clamp_i32(out_val, 0, 255);
            }
        }
    }
}

/* Depthwise 2D Convolution (INT8, per-channel) */
static void depthwise_conv2d_int8(
    const uint8_t *input, uint8_t *output,
    const int8_t *weight, const int32_t *bias, const float *w_scale,
    float in_scale, uint8_t in_zp,
    float out_scale, uint8_t out_zp,
    int channels, int sp_in, int sp_out,
    int kh, int kw, int stride, int pad
) {
    int ch, oh, ow, fh, fw;
    for (ch = 0; ch < channels; ch++) {
        float M = (in_scale * w_scale[ch]) / out_scale;
        for (oh = 0; oh < sp_out; oh++) {
            for (ow = 0; ow < sp_out; ow++) {
                int32_t acc = bias[ch];
                for (fh = 0; fh < kh; fh++) {
                    for (fw = 0; fw < kw; fw++) {
                        int ih = oh * stride - pad + fh;
                        int iw = ow * stride - pad + fw;
                        int32_t x_val = 0;
                        if (ih >= 0 && ih < sp_in && iw >= 0 && iw < sp_in) {
                            x_val = (int32_t)input[ch * sp_in * sp_in + ih * sp_in + iw] - (int32_t)in_zp;
                        } else {
                            x_val = -(int32_t)in_zp;
                        }
                        int w_idx = ch * (kh * kw) + fh * kw + fw;
                        int32_t w_val = (int32_t)read_weight(weight, w_idx);
                        acc += x_val * w_val;
                    }
                }
                int32_t out_val = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;
                output[ch * sp_out * sp_out + oh * sp_out + ow] = (uint8_t)clamp_i32(out_val, 0, 255);
            }
        }
    }
}

/* Quantized element-wise add */
static void qadd_int8(
    const uint8_t *a, const uint8_t *b, uint8_t *out,
    float a_scale, uint8_t a_zp,
    float b_scale, uint8_t b_zp,
    float out_scale, uint8_t out_zp,
    int size
) {
    float a_ratio = a_scale / out_scale;
    float b_ratio = b_scale / out_scale;
    int i;
    for (i = 0; i < size; i++) {
        float val = ((float)a[i] - (float)a_zp) * a_ratio
                  + ((float)b[i] - (float)b_zp) * b_ratio
                  + (float)out_zp;
        int32_t ival = (int32_t)(val + 0.5f);
        out[i] = (uint8_t)clamp_i32(ival, 0, 255);
    }
}

/* Global average pool: uint8 -> float per channel */
static void global_avg_pool_uint8(
    const uint8_t *input, float *output,
    float scale, uint8_t zp,
    int channels, int h, int w
) {
    int c, y, x;
    int hw = h * w;
    for (c = 0; c < channels; c++) {
        int32_t sum = 0;
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                sum += (int32_t)input[c * hw + y * w + x];
            }
        }
        /* Dequantize the mean */
        float mean_q = (float)sum / (float)hw;
        output[c] = (mean_q - (float)zp) * scale;
    }
}

/* Dense layer: quantized -> float output */
static void dense_int8(
    const float *input_float,
    float *output,
    const int8_t *weight, const int32_t *bias, const float *w_scale,
    float in_scale, uint8_t in_zp,
    float out_scale, uint8_t out_zp,
    int in_features, int out_features
) {
    int i, j;
    /* Quantize float input to uint8 */
    static uint8_t q_in[DENSE_IN];
    for (i = 0; i < in_features; i++) {
        float qf = input_float[i] / in_scale + (float)in_zp;
        q_in[i] = (uint8_t)clamp_i32((int32_t)(qf + 0.5f), 0, 255);
    }

    /* Matrix multiply: weight is [out_features, in_features] (transB=1 in QGemm) */
    for (i = 0; i < out_features; i++) {
        int32_t acc = bias[i];
        float M = (in_scale * w_scale[i]) / out_scale;
        for (j = 0; j < in_features; j++) {
            int32_t x_val = (int32_t)q_in[j] - (int32_t)in_zp;
            int32_t w_val = (int32_t)read_weight(weight, i * in_features + j);
            acc += x_val * w_val;
        }
        /* Dequantize output to float */
        int32_t out_q = (int32_t)(acc * M + 0.5f) + (int32_t)out_zp;
        output[i] = ((float)out_q - (float)out_zp) * out_scale;
    }
}

/* Softmax */
static void softmax(float *x, int n) {
    int i;
    float max_val = x[0];
    for (i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/* Main inference function */
void predict(const uint8_t *input_quant, float *output) {
    uint8_t *cur_in  = buf_a;
    uint8_t *cur_out = buf_b;
    uint8_t *tmp;

    /* Copy quantized input to working buffer */
    memcpy(cur_in, input_quant, INPUT_SIZE);

    /* Layer 0: Stem Conv 3->16 3x3 s=1 p=1 */
    conv2d_int8(cur_in, cur_out,
        conv0_weight, conv0_bias, conv0_w_scale,
        conv0_in_scale, conv0_in_zp, conv0_out_scale, conv0_out_zp,
        L0_IN_C, L0_OUT_C, L0_SP_IN, L0_SP_OUT,
        L0_KH, L0_KW, L0_STRIDE, L0_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Save stem output for skip connection */
    memcpy(residual_buf, cur_in, L0_OUT_C * L0_SP_OUT * L0_SP_OUT);

    /* Layer 1: MBConv1 Depthwise 16ch 3x3 s=1 p=1 */
    depthwise_conv2d_int8(cur_in, cur_out,
        conv1_weight, conv1_bias, conv1_w_scale,
        conv1_in_scale, conv1_in_zp, conv1_out_scale, conv1_out_zp,
        L1_IN_C, L1_SP_IN, L1_SP_OUT,
        L1_KH, L1_KW, L1_STRIDE, L1_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 2: MBConv1 Pointwise 16->16 1x1 */
    conv2d_int8(cur_in, cur_out,
        conv2_weight, conv2_bias, conv2_w_scale,
        conv2_in_scale, conv2_in_zp, conv2_out_scale, conv2_out_zp,
        L2_IN_C, L2_OUT_C, L2_SP_IN, L2_SP_OUT,
        L2_KH, L2_KW, L2_STRIDE, L2_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Residual add: stem_output + mbconv1_output */
    qadd_int8(residual_buf, cur_in, cur_out,
        add_a_scale, add_a_zp,
        add_b_scale, add_b_zp,
        add_out_scale, add_out_zp,
        16 * 32 * 32);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 3: Conv 16->32 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv3_weight, conv3_bias, conv3_w_scale,
        conv3_in_scale, conv3_in_zp, conv3_out_scale, conv3_out_zp,
        L3_IN_C, L3_OUT_C, L3_SP_IN, L3_SP_OUT,
        L3_KH, L3_KW, L3_STRIDE, L3_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 4: Depthwise 32->32 3x3 s=2 p=1 */
    depthwise_conv2d_int8(cur_in, cur_out,
        conv4_weight, conv4_bias, conv4_w_scale,
        conv4_in_scale, conv4_in_zp, conv4_out_scale, conv4_out_zp,
        L4_IN_C, L4_SP_IN, L4_SP_OUT,
        L4_KH, L4_KW, L4_STRIDE, L4_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 5: Conv 32->24 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv5_weight, conv5_bias, conv5_w_scale,
        conv5_in_scale, conv5_in_zp, conv5_out_scale, conv5_out_zp,
        L5_IN_C, L5_OUT_C, L5_SP_IN, L5_SP_OUT,
        L5_KH, L5_KW, L5_STRIDE, L5_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 6: Conv 24->48 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv6_weight, conv6_bias, conv6_w_scale,
        conv6_in_scale, conv6_in_zp, conv6_out_scale, conv6_out_zp,
        L6_IN_C, L6_OUT_C, L6_SP_IN, L6_SP_OUT,
        L6_KH, L6_KW, L6_STRIDE, L6_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 7: Depthwise 48->48 3x3 s=2 p=1 */
    depthwise_conv2d_int8(cur_in, cur_out,
        conv7_weight, conv7_bias, conv7_w_scale,
        conv7_in_scale, conv7_in_zp, conv7_out_scale, conv7_out_zp,
        L7_IN_C, L7_SP_IN, L7_SP_OUT,
        L7_KH, L7_KW, L7_STRIDE, L7_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 8: Conv 48->32 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv8_weight, conv8_bias, conv8_w_scale,
        conv8_in_scale, conv8_in_zp, conv8_out_scale, conv8_out_zp,
        L8_IN_C, L8_OUT_C, L8_SP_IN, L8_SP_OUT,
        L8_KH, L8_KW, L8_STRIDE, L8_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 9: Conv 32->64 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv9_weight, conv9_bias, conv9_w_scale,
        conv9_in_scale, conv9_in_zp, conv9_out_scale, conv9_out_zp,
        L9_IN_C, L9_OUT_C, L9_SP_IN, L9_SP_OUT,
        L9_KH, L9_KW, L9_STRIDE, L9_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 10: Depthwise 64->64 3x3 s=2 p=1 */
    depthwise_conv2d_int8(cur_in, cur_out,
        conv10_weight, conv10_bias, conv10_w_scale,
        conv10_in_scale, conv10_in_zp, conv10_out_scale, conv10_out_zp,
        L10_IN_C, L10_SP_IN, L10_SP_OUT,
        L10_KH, L10_KW, L10_STRIDE, L10_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 11: Conv 64->64 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv11_weight, conv11_bias, conv11_w_scale,
        conv11_in_scale, conv11_in_zp, conv11_out_scale, conv11_out_zp,
        L11_IN_C, L11_OUT_C, L11_SP_IN, L11_SP_OUT,
        L11_KH, L11_KW, L11_STRIDE, L11_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Layer 12: Conv 64->128 1x1 s=1 p=0 */
    conv2d_int8(cur_in, cur_out,
        conv12_weight, conv12_bias, conv12_w_scale,
        conv12_in_scale, conv12_in_zp, conv12_out_scale, conv12_out_zp,
        L12_IN_C, L12_OUT_C, L12_SP_IN, L12_SP_OUT,
        L12_KH, L12_KW, L12_STRIDE, L12_PAD);
    tmp = cur_in; cur_in = cur_out; cur_out = tmp;

    /* Global Average Pool: 128x4x4 -> 128 */
    static float pool_out[POOL_C];
    global_avg_pool_uint8(cur_in, pool_out,
        conv12_out_scale, conv12_out_zp,
        POOL_C, POOL_H, POOL_W);

    /* Dense: 128 -> 10 */
    dense_int8(pool_out, output,
        dense_weight, dense_bias, dense_w_scale,
        dense_in_scale, dense_in_zp,
        dense_out_scale, dense_out_zp,
        DENSE_IN, DENSE_OUT);

    /* Softmax */
    softmax(output, NUM_CLASSES);
}

