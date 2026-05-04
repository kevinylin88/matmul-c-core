//以下代码由Codex撰写
//性能表现较差（后面有空时把multiply_neon和neon kernel逻辑重新手写改下）
#ifdef __aarch64__
#include "multiply.h"

#include <arm_neon.h>

#define NEON_PANEL_COLS 16
#define NEON_VECTORS_PER_PANEL 4

#define DEFINE_NEON_KERNEL_X16(ROWS) \ //'\'这是宏的续航符,告诉preprecessor，下一行依旧是宏的一部分
void neon_kernel_##ROWS##x16( \
    struct Matrix mat1, \
    struct Matrix mat2, \
    struct Matrix mat3, \
    int id, \
    int jd, \
    size_t i_start, \
    size_t j_start, \
    size_t valid_k \
){ \
    //crows的每一个元素表示第i行的第一个指针地址
    float *c_rows[(ROWS)]; \
    //累加器，一开始先存储a的数据
    float32x4_t acc[(ROWS)][NEON_VECTORS_PER_PANEL]; \ //一个float是32，一个panel是128，所以一个panel的行装有4个
    size_t panel = (size_t)jd / NEON_PANEL_COLS; \ 
    float *panel_ptr = mat2.data + panel * valid_k * NEON_PANEL_COLS; \
    \
    for(size_t row = 0; row < (ROWS); row++){ \
        c_rows[row] = mat3.data + (i_start + (size_t)id + row) * mat3.cols + j_start + (size_t)jd; \
        for(size_t vec = 0; vec < NEON_VECTORS_PER_PANEL; vec++){ \
            //连续读取4个float，放到一个NEON向量寄存器当中
            acc[row][vec] = vld1q_f32(c_rows[row] + vec * 4); \
        } \
    } \
    \
    for(size_t k = 0; k < valid_k; k++){ \
        float *b_ptr = panel_ptr + k * NEON_PANEL_COLS; \
        //b的第k行的前16个
        float32x4_t b[NEON_VECTORS_PER_PANEL] = { \
            vld1q_f32(b_ptr), \
            vld1q_f32(b_ptr + 4), \
            vld1q_f32(b_ptr + 8), \
            vld1q_f32(b_ptr + 12) \
        }; \
        \
        for(size_t row = 0; row < (ROWS); row++){ \
            float a_value = mat1.data[((size_t)id + row) * mat1.cols + k]; \
            //广播：vdup的意思应当是vector duplicate
            float32x4_t a = vdupq_n_f32(a_value); \
            for(size_t vec = 0; vec < NEON_VECTORS_PER_PANEL; vec++){ \
                //累加a*b + c
                acc[row][vec] = vfmaq_f32(acc[row][vec], a, b[vec]); \
            } \
        } \
    } \
    \
    //写回
    for(size_t row = 0; row < (ROWS); row++){ \
        for(size_t vec = 0; vec < NEON_VECTORS_PER_PANEL; vec++){ \
            vst1q_f32(c_rows[row] + vec * 4, acc[row][vec]); \
        } \
    } \
}

DEFINE_NEON_KERNEL_X16(1)
DEFINE_NEON_KERNEL_X16(2)
DEFINE_NEON_KERNEL_X16(4)
DEFINE_NEON_KERNEL_X16(6)

void neon_kernel_panelb_tail(
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
            float a = mat1.data[((size_t)id + row) * mat1.cols + k];
            for(size_t col = 0; col < cols; col++){
                size_t abs_col = (size_t)jd + col;
                size_t panel = abs_col / NEON_PANEL_COLS;
                size_t inner = abs_col % NEON_PANEL_COLS;
                float b = mat2.data[panel * valid_k * NEON_PANEL_COLS + k * NEON_PANEL_COLS + inner];
                mat3_row[col] += a * b;
            }
        }
    }
}

#endif
