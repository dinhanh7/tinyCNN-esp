/* AUTO-GENERATED - DO NOT EDIT */
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

/* ---- Model dimensions ---- */
#define INPUT_C     3
#define INPUT_H     32
#define INPUT_W     32
#define INPUT_SIZE  (INPUT_C * INPUT_H * INPUT_W)   /* 3072 */
#define NUM_CLASSES 10
#define NUM_CONV_LAYERS 13
#define MAX_BUF_SIZE 32768
#define RESIDUAL_SIZE 16384

/* ---- Per-layer dimensions ---- */
#define L0_IN_C   3
#define L0_OUT_C  16
#define L0_KH     3
#define L0_KW     3
#define L0_STRIDE 1
#define L0_PAD    1
#define L0_GROUP  1
#define L0_SP_IN  32
#define L0_SP_OUT 32
#define L0_DW     0

#define L1_IN_C   16
#define L1_OUT_C  16
#define L1_KH     3
#define L1_KW     3
#define L1_STRIDE 1
#define L1_PAD    1
#define L1_GROUP  16
#define L1_SP_IN  32
#define L1_SP_OUT 32
#define L1_DW     1

#define L2_IN_C   16
#define L2_OUT_C  16
#define L2_KH     1
#define L2_KW     1
#define L2_STRIDE 1
#define L2_PAD    0
#define L2_GROUP  1
#define L2_SP_IN  32
#define L2_SP_OUT 32
#define L2_DW     0

#define L3_IN_C   16
#define L3_OUT_C  32
#define L3_KH     1
#define L3_KW     1
#define L3_STRIDE 1
#define L3_PAD    0
#define L3_GROUP  1
#define L3_SP_IN  32
#define L3_SP_OUT 32
#define L3_DW     0

#define L4_IN_C   32
#define L4_OUT_C  32
#define L4_KH     3
#define L4_KW     3
#define L4_STRIDE 2
#define L4_PAD    1
#define L4_GROUP  32
#define L4_SP_IN  32
#define L4_SP_OUT 16
#define L4_DW     1

#define L5_IN_C   32
#define L5_OUT_C  24
#define L5_KH     1
#define L5_KW     1
#define L5_STRIDE 1
#define L5_PAD    0
#define L5_GROUP  1
#define L5_SP_IN  16
#define L5_SP_OUT 16
#define L5_DW     0

#define L6_IN_C   24
#define L6_OUT_C  48
#define L6_KH     1
#define L6_KW     1
#define L6_STRIDE 1
#define L6_PAD    0
#define L6_GROUP  1
#define L6_SP_IN  16
#define L6_SP_OUT 16
#define L6_DW     0

#define L7_IN_C   48
#define L7_OUT_C  48
#define L7_KH     3
#define L7_KW     3
#define L7_STRIDE 2
#define L7_PAD    1
#define L7_GROUP  48
#define L7_SP_IN  16
#define L7_SP_OUT 8
#define L7_DW     1

#define L8_IN_C   48
#define L8_OUT_C  32
#define L8_KH     1
#define L8_KW     1
#define L8_STRIDE 1
#define L8_PAD    0
#define L8_GROUP  1
#define L8_SP_IN  8
#define L8_SP_OUT 8
#define L8_DW     0

#define L9_IN_C   32
#define L9_OUT_C  64
#define L9_KH     1
#define L9_KW     1
#define L9_STRIDE 1
#define L9_PAD    0
#define L9_GROUP  1
#define L9_SP_IN  8
#define L9_SP_OUT 8
#define L9_DW     0

#define L10_IN_C   64
#define L10_OUT_C  64
#define L10_KH     3
#define L10_KW     3
#define L10_STRIDE 2
#define L10_PAD    1
#define L10_GROUP  64
#define L10_SP_IN  8
#define L10_SP_OUT 4
#define L10_DW     1

#define L11_IN_C   64
#define L11_OUT_C  64
#define L11_KH     1
#define L11_KW     1
#define L11_STRIDE 1
#define L11_PAD    0
#define L11_GROUP  1
#define L11_SP_IN  4
#define L11_SP_OUT 4
#define L11_DW     0

#define L12_IN_C   64
#define L12_OUT_C  128
#define L12_KH     1
#define L12_KW     1
#define L12_STRIDE 1
#define L12_PAD    0
#define L12_GROUP  1
#define L12_SP_IN  4
#define L12_SP_OUT 4
#define L12_DW     0

#define DENSE_IN   128
#define DENSE_OUT  10

#define POOL_C   128
#define POOL_H   4
#define POOL_W   4

/* ---- Function prototypes ---- */
#ifdef __cplusplus
extern "C" {
#endif

void predict(const uint8_t *input_quant, float *output);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
