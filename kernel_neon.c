#ifdef __aarch64__
#include "multiply.h"

#include <arm_neon.h>

void neon_kernel_1x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    float *mat3_pos0 = mat3.data + (i_start + (size_t)id) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos1 = mat3_pos0 + 4;
    float *mat3_pos2 = mat3_pos0 + 8;
    float *mat3_pos3 = mat3_pos0 + 12;

    float32x4_t acc0 = vld1q_f32(mat3_pos0);
    float32x4_t acc1 = vld1q_f32(mat3_pos1);
    float32x4_t acc2 = vld1q_f32(mat3_pos2);
    float32x4_t acc3 = vld1q_f32(mat3_pos3);

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);
        float32x4_t b2 = vld1q_f32(b_ptr + 8);
        float32x4_t b3 = vld1q_f32(b_ptr + 12);
        float32x4_t a = vdupq_n_f32(*(mat1.data + (size_t)id * mat1.cols + k));

        acc0 = vfmaq_f32(acc0, a, b0);
        acc1 = vfmaq_f32(acc1, a, b1);
        acc2 = vfmaq_f32(acc2, a, b2);
        acc3 = vfmaq_f32(acc3, a, b3);
    }

    vst1q_f32(mat3_pos0, acc0);
    vst1q_f32(mat3_pos1, acc1);
    vst1q_f32(mat3_pos2, acc2);
    vst1q_f32(mat3_pos3, acc3);
}

void neon_kernel_2x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    float *mat3_pos0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos1 = mat3_pos0 + 4;
    float *mat3_pos2 = mat3_pos0 + 8;
    float *mat3_pos3 = mat3_pos0 + 12;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3_pos4 + 4;
    float *mat3_pos6 = mat3_pos4 + 8;
    float *mat3_pos7 = mat3_pos4 + 12;

    float32x4_t acc0 = vld1q_f32(mat3_pos0);
    float32x4_t acc1 = vld1q_f32(mat3_pos1);
    float32x4_t acc2 = vld1q_f32(mat3_pos2);
    float32x4_t acc3 = vld1q_f32(mat3_pos3);
    float32x4_t acc4 = vld1q_f32(mat3_pos4);
    float32x4_t acc5 = vld1q_f32(mat3_pos5);
    float32x4_t acc6 = vld1q_f32(mat3_pos6);
    float32x4_t acc7 = vld1q_f32(mat3_pos7);

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);
        float32x4_t b2 = vld1q_f32(b_ptr + 8);
        float32x4_t b3 = vld1q_f32(b_ptr + 12);

        float32x4_t a = vdupq_n_f32(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = vfmaq_f32(acc0, a, b0);
        acc1 = vfmaq_f32(acc1, a, b1);
        acc2 = vfmaq_f32(acc2, a, b2);
        acc3 = vfmaq_f32(acc3, a, b3);

        a = vdupq_n_f32(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc4 = vfmaq_f32(acc4, a, b0);
        acc5 = vfmaq_f32(acc5, a, b1);
        acc6 = vfmaq_f32(acc6, a, b2);
        acc7 = vfmaq_f32(acc7, a, b3);
    }

    vst1q_f32(mat3_pos0, acc0);
    vst1q_f32(mat3_pos1, acc1);
    vst1q_f32(mat3_pos2, acc2);
    vst1q_f32(mat3_pos3, acc3);
    vst1q_f32(mat3_pos4, acc4);
    vst1q_f32(mat3_pos5, acc5);
    vst1q_f32(mat3_pos6, acc6);
    vst1q_f32(mat3_pos7, acc7);
}

void neon_kernel_4x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    float *row0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + j_start + (size_t)jd;
    float *row1 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *row2 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *row3 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;

    float32x4_t acc[4][4];
    float *rows[4] = {row0, row1, row2, row3};
    for(size_t r = 0; r < 4; r++){
        for(size_t v = 0; v < 4; v++){
            acc[r][v] = vld1q_f32(rows[r] + v * 4);
        }
    }

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        float32x4_t b[4] = {
            vld1q_f32(b_ptr),
            vld1q_f32(b_ptr + 4),
            vld1q_f32(b_ptr + 8),
            vld1q_f32(b_ptr + 12)
        };
        for(size_t r = 0; r < 4; r++){
            float32x4_t a = vdupq_n_f32(*(mat1.data + ((size_t)id + r) * mat1.cols + k));
            for(size_t v = 0; v < 4; v++){
                acc[r][v] = vfmaq_f32(acc[r][v], a, b[v]);
            }
        }
    }

    for(size_t r = 0; r < 4; r++){
        for(size_t v = 0; v < 4; v++){
            vst1q_f32(rows[r] + v * 4, acc[r][v]);
        }
    }
}

void neon_kernel_6x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    float *rows[6];
    float32x4_t acc[6][4];
    for(size_t r = 0; r < 6; r++){
        rows[r] = mat3.data + (i_start + (size_t)id + r) * mat3.cols + j_start + (size_t)jd;
        for(size_t v = 0; v < 4; v++){
            acc[r][v] = vld1q_f32(rows[r] + v * 4);
        }
    }

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        float32x4_t b[4] = {
            vld1q_f32(b_ptr),
            vld1q_f32(b_ptr + 4),
            vld1q_f32(b_ptr + 8),
            vld1q_f32(b_ptr + 12)
        };
        for(size_t r = 0; r < 6; r++){
            float32x4_t a = vdupq_n_f32(*(mat1.data + ((size_t)id + r) * mat1.cols + k));
            for(size_t v = 0; v < 4; v++){
                acc[r][v] = vfmaq_f32(acc[r][v], a, b[v]);
            }
        }
    }

    for(size_t r = 0; r < 6; r++){
        for(size_t v = 0; v < 4; v++){
            vst1q_f32(rows[r] + v * 4, acc[r][v]);
        }
    }
}

static void scalar_kernel_row_major(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k,
    size_t rows,
    size_t cols
){
    for(size_t row = 0; row < rows; row++){
        float *mat3_row = mat3.data + (i_start + (size_t)id + row) * mat3.cols + j_start + (size_t)jd;
        for(size_t k = 0; k < valid_k; k++){
            float a = *(mat1.data + ((size_t)id + row) * mat1.cols + k);
            float *mat2_row = mat2.data + k * mat2.cols + (size_t)jd;
            for(size_t col = 0; col < cols; col++){
                *(mat3_row + col) += a * *(mat2_row + col);
            }
        }
    }
}

static void scalar_kernel_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k,
    size_t rows,
    size_t cols
){
    for(size_t row = 0; row < rows; row++){
        float *mat3_row = mat3.data + (i_start + (size_t)id + row) * mat3.cols + j_start + (size_t)jd;
        for(size_t k = 0; k < valid_k; k++){
            float a = *(mat1.data + ((size_t)id + row) * mat1.cols + k);
            for(size_t col = 0; col < cols; col++){
                size_t abs_col = (size_t)jd + col;
                size_t panel = abs_col / NR;
                size_t inner = abs_col % NR;
                float b = *(mat2.data + panel * valid_k * NR + k * NR + inner);
                *(mat3_row + col) += a * b;
            }
        }
    }
}

void avx_kernel_1x8(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 1, 8);
}

void avx_kernel_2x8(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 2, 8);
}

void avx_kernel_4x8(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 4, 8);
}

void avx_kernel_6x8(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 6, 8);
}

void avx_kernel_2x16(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 2, 16);
}

void avx_kernel_6x16(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 6, 16);
}

void avx512_kernel_1x32(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 1, 32);
}

void avx512_kernel_2x32(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 2, 32);
}

void avx512_kernel_4x32(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 4, 32);
}

void avx512_kernel_6x32(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_row_major(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 6, 32);
}

void avx512_kernel_1x32_panelb(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_panelb(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 1, 32);
}

void avx512_kernel_2x32_panelb(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_panelb(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 2, 32);
}

void avx512_kernel_4x32_panelb(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_panelb(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 4, 32);
}

void avx512_kernel_6x32_panelb(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, int id, int jd, size_t i_start, size_t j_start, size_t valid_k){
    scalar_kernel_panelb(mat1, mat2, mat3, id, jd, i_start, j_start, valid_k, 6, 32);
}
#endif
