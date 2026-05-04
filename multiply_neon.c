#ifdef __aarch64__
#include "multiply.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_matrix_neon(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(mat1.data == NULL || mat2.data == NULL || mat3.data == NULL){
        fprintf(stderr, "Error: matrix data pointer is NULL.\n");
        return -1;
    }
    if(mat1.rows == 0 || mat1.cols == 0 || mat2.rows == 0 || mat2.cols == 0 || mat3.rows == 0 || mat3.cols == 0){
        fprintf(stderr, "Error: matrix dimensions must be greater than 0.\n");
        return -1;
    }
    if(mat1.cols != mat2.rows){
        fprintf(stderr, "Error: matrix dimensions do not match for multiplication.\n");
        return -1;
    }
    if(mat3.rows != mat1.rows || mat3.cols != mat2.cols){
        fprintf(stderr, "Error: output matrix dimensions are incorrect.\n");
        return -1;
    }
    return 0;
}

static void *aligned_alloc64_floats_neon(size_t count){
    size_t bytes = count * sizeof(float);
    size_t aligned_bytes = ((bytes + 63) / 64) * 64;
    return aligned_alloc(64, aligned_bytes);
}

void matmul_v8_neon_omp(
    struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, size_t ii, size_t jj, size_t kk
){
    if(check_matrix_neon(mat1, mat2, mat3) == -1){exit(-1);}
    if(ii == 0 || jj == 0 || kk == 0){
        fprintf(stderr, "Error: block parameters must be greater than 0.\n");
        exit(-1);
    }

    struct Matrix mat2_block;
    size_t mat2_block_panels = (jj + NR - 1) / NR;
    mat2_block.data = aligned_alloc64_floats_neon(mat2_block_panels * kk * NR);

    if(mat2_block.data == NULL){
        fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n", BLOCK_LEAP, BLOCK_LEAP);
        free(mat2_block.data);
        exit(-1);
    }

    #pragma omp parallel
    {
        struct Matrix mat1_block;
        mat1_block.data = aligned_alloc64_floats_neon(ii * kk);

        if(mat1_block.data == NULL){
            fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n", BLOCK_LEAP, BLOCK_LEAP);
            free(mat1_block.data);
            exit(-1);
        }

        for(size_t j = 0; j < mat2.cols; j+=jj){
            size_t j_start = j, j_end = MIN(j + jj, mat2.cols);

            for(size_t k = 0; k < mat1.cols; k+=kk){
                size_t k_start = k, k_end = MIN(k + kk, mat1.cols);

                #pragma omp single
                {
                    mat2_block.rows = k_end - k_start;
                    mat2_block.cols = j_end - j_start;

                    size_t current_panels = ((size_t)mat2_block.cols + NR - 1) / NR;
                    for(size_t panel = 0; panel < current_panels; panel++){
                        size_t panel_cols = MIN((size_t)NR, mat2_block.cols - panel * NR);
                        for(size_t kd = 0; kd < mat2_block.rows; kd++){
                            float *dst = mat2_block.data + panel * mat2_block.rows * NR + kd * NR;
                            float *src = mat2.data + (k_start + kd) * mat2.cols + j_start + panel * NR;
                            for(size_t inner = 0; inner < panel_cols; inner++){
                                *(dst + inner) = *(src + inner);
                            }
                        }
                    }
                }

                #pragma omp for
                for(size_t i = 0; i < mat1.rows; i+=ii){
                    size_t i_start = i, i_end = MIN(i + ii, mat1.rows);

                    mat1_block.rows = i_end - i_start;
                    mat1_block.cols = k_end - k_start;

                    for(size_t id = 0; id < mat1_block.rows; id++){
                        for(size_t kd = 0; kd < mat1_block.cols; kd++){
                            *(mat1_block.data + id * mat1_block.cols + kd) =
                            *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                        }
                    }

                    for(size_t id = 0; id < mat1_block.rows;){
                        if(id + 6 <= mat1_block.rows){
                            size_t jd = 0;
                            for(; jd + 16 <= mat2_block.cols; jd += 16){
                                neon_kernel_6x16(mat1_block, mat2_block, mat3, (int)id, (int)jd, i_start, j_start, mat1_block.cols);
                            }
                            for(size_t row = 0; row < 6; row++){
                                float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                                for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                    float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                    for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                        size_t panel = tail_jd / NR;
                                        size_t inner = tail_jd % NR;
                                        float b = *(mat2_block.data + panel * mat2_block.rows * NR + kd * NR + inner);
                                        *(mat3_row + tail_jd) += a * b;
                                    }
                                }
                            }
                            id += 6;
                        }
                        else if(id + 4 <= mat1_block.rows){
                            size_t jd = 0;
                            for(; jd + 16 <= mat2_block.cols; jd += 16){
                                neon_kernel_4x16(mat1_block, mat2_block, mat3, (int)id, (int)jd, i_start, j_start, mat1_block.cols);
                            }
                            for(size_t row = 0; row < 4; row++){
                                float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                                for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                    float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                    for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                        size_t panel = tail_jd / NR;
                                        size_t inner = tail_jd % NR;
                                        float b = *(mat2_block.data + panel * mat2_block.rows * NR + kd * NR + inner);
                                        *(mat3_row + tail_jd) += a * b;
                                    }
                                }
                            }
                            id += 4;
                        }
                        else if(id + 2 <= mat1_block.rows){
                            size_t jd = 0;
                            for(; jd + 16 <= mat2_block.cols; jd += 16){
                                neon_kernel_2x16(mat1_block, mat2_block, mat3, (int)id, (int)jd, i_start, j_start, mat1_block.cols);
                            }
                            for(size_t row = 0; row < 2; row++){
                                float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                                for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                    float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                    for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                        size_t panel = tail_jd / NR;
                                        size_t inner = tail_jd % NR;
                                        float b = *(mat2_block.data + panel * mat2_block.rows * NR + kd * NR + inner);
                                        *(mat3_row + tail_jd) += a * b;
                                    }
                                }
                            }
                            id += 2;
                        }
                        else{
                            size_t jd = 0;
                            for(; jd + 16 <= mat2_block.cols; jd += 16){
                                neon_kernel_1x16(mat1_block, mat2_block, mat3, (int)id, (int)jd, i_start, j_start, mat1_block.cols);
                            }
                            float *mat3_row = mat3.data + (i_start + id) * mat3.cols + j_start;
                            for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                float a = *(mat1_block.data + id * mat1_block.cols + kd);
                                for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                    size_t panel = tail_jd / NR;
                                    size_t inner = tail_jd % NR;
                                    float b = *(mat2_block.data + panel * mat2_block.rows * NR + kd * NR + inner);
                                    *(mat3_row + tail_jd) += a * b;
                                }
                            }
                            id++;
                        }
                    }
                }
            }
        }
        free(mat1_block.data);
    }

    free(mat2_block.data);
}
#endif
