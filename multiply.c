#include "multiply.h"
#include <stddef.h>

static int check_matrix(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
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
        exit(-1);
    }
    return 0;
}

int set_memory_limit(size_t bytes){
    struct rlimit limit;//储存soft/hard limit的结构体
    limit.rlim_cur = (rlim_t)bytes;//转换到rlim_t
    limit.rlim_max = (rlim_t)bytes;
    return setrlimit(RLIMIT_AS, &limit);
}

void multiply_plain(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}//check负责找问题，这里负责退出

    for(size_t i = 0; i < mat1.rows; i++){//选定行
        for(size_t j = 0; j < mat2.cols; j++){//选定列
            float sum = 0.0f;
            for(size_t k = 0; k < mat1.cols; k++){
                sum += *(mat1.data + i * mat1.cols + k) * *(mat2.data + k * mat2.cols + j);
            }
            *(mat3.data + i * mat3.cols + j) = sum;//mat3是i,j
        }
    }
}


void matmul_v2_ikj(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}

    for(size_t i = 0; i < mat1.rows; i++){
        for(size_t k = 0; k < mat1.cols; k++){
            for(size_t j = 0; j < mat2.cols; j++){
                *(mat3.data+i*mat3.cols+j) += *(mat1.data+i*mat1.cols+k) * *(mat2.data + k*mat2.cols + j);
            }
        }
    }
}//ikj顺序贴合内存访问方式

void matmul_v3_block(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}

    //挑选分块矩阵
    for (size_t i = 0; i < mat1.rows; i += BLOCK_LEAP){
        for(size_t j = 0; j < mat2.cols; j += BLOCK_LEAP){
            for(size_t k = 0; k < mat1.cols; k = k + BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);//要考虑边界
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);
                //小矩阵内乘法
                for(size_t id = i_start; id < i_end; id++){
                    for(size_t kd = k_start; kd < k_end; kd++){
                        for(size_t jd = j_start; jd < j_end; jd++){
                            *(mat3.data + id*mat3.cols + jd) += *(mat1.data + id*mat1.cols + kd) * *(mat2.data + kd*mat2.cols + jd);
                        }
                    }
                }
            }
        }
    }
}

//加入分块和AVX2指令的版本
void matmul_v4_avx2(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}

    for(size_t i = 0; i < mat1.rows; i+=BLOCK_LEAP){
        for(size_t j = 0; j < mat2.cols; j+=BLOCK_LEAP){
            //做pack
            struct Matrix mat1_block; struct Matrix mat2_block;
            mat1_block.data = malloc(sizeof(float) * BLOCK_LEAP * BLOCK_LEAP);
            mat2_block.data = malloc(sizeof(float) * BLOCK_LEAP * BLOCK_LEAP);

            if(mat1_block.data == NULL || mat2_block.data == NULL){//错误处理
                fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n", BLOCK_LEAP, BLOCK_LEAP);
                free(mat1_block.data);
                free(mat2_block.data);
                exit(-1);
            }

            for(size_t k = 0; k < mat1.cols; k+=BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);//要考虑边界
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);
                
                mat1_block.rows = i_end - i_start; mat1_block.cols = k_end - k_start;
                mat2_block.rows = k_end - k_start; mat2_block.cols = j_end - j_start;

                //pack mat1,这里虽似乎多此一举，但用一个双重循环换来未来上百次的内存连续访问
                //还是非常值的
                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                        *(mat1_block.data + id*mat1_block.cols + kd) = 
                        *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                    }
                }

                //pack mat2
                for(size_t kd = 0; kd < mat2_block.rows; kd++){
                    for(size_t jd = 0; jd < mat2_block.cols; jd++){
                        *(mat2_block.data + kd * mat2_block.cols + jd) = 
                        *(mat2.data + (k_start + kd) * mat2.cols + j_start + jd);
                    }
                }

                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t jd = 0; jd < mat2_block.cols; jd += SIMD_LEAP){
                        if(jd + SIMD_LEAP <= mat2_block.cols){
                            avx_kernel_1x8(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                (int)id, 
                                (int)jd, 
                                i_start, 
                                j_start,
                                mat1_block.cols);
                        }else{
                            for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                float sum = 0.0f;
                                for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                    sum += *(mat1_block.data + id * mat1_block.cols + kd) *
                                           *(mat2_block.data + kd * mat2_block.cols + tail_jd);
                                }
                                *(mat3.data + (i_start + id) * mat3.cols + j_start + tail_jd) += sum;
                            }
                        }
                    }
                }
            }
            free(mat1_block.data);
            free(mat2_block.data);
        }
    }
}

//加入OpenMP多线程和预取指令的版本
void matmul_v5_openmp(
    struct Matrix mat1, struct Matrix mat2, struct Matrix mat3
){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}//入口检查

    #pragma omp parallel for collapse(2)
    for(size_t i = 0; i < mat1.rows; i+=BLOCK_LEAP){
        for(size_t j = 0; j < mat2.cols; j+=BLOCK_LEAP){
            //做pack
            struct Matrix mat1_block; struct Matrix mat2_block;

            mat1_block.data = malloc(sizeof(float) * BLOCK_LEAP * BLOCK_LEAP);
            mat2_block.data = malloc(sizeof(float) * BLOCK_LEAP * BLOCK_LEAP);

            if(mat1_block.data == NULL || mat2_block.data == NULL){//错误处理
                    fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n",BLOCK_LEAP, BLOCK_LEAP);
                    free(mat1_block.data);
                    free(mat2_block.data);
                    exit(-1);
                }

            for(size_t k = 0; k < mat1.cols; k+=BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);//要考虑边界
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);

                mat1_block.rows = i_end - i_start; mat1_block.cols = k_end - k_start;
                mat2_block.rows = k_end - k_start; mat2_block.cols = j_end - j_start;//无论如何，row/col永远比BLOCK_LEAP小


                //pack mat1,这里虽似乎多此一举，但用一个双重循环换来未来上百次的内存连续访问
                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                        *(mat1_block.data + id*mat1_block.cols + kd) = 
                        *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                    }
                }

                //pack mat2
                for(size_t kd = 0; kd < mat2_block.rows; kd++){
                    for(size_t jd = 0; jd < mat2_block.cols; jd++){
                        *(mat2_block.data + kd * mat2_block.cols + jd) = 
                        *(mat2.data + (k_start + kd) * mat2.cols + j_start + jd);
                    }
                }
                for(size_t id = 0; id < mat1_block.rows; id += 4){
                    for(size_t jd = 0; jd < mat2_block.cols; jd += SIMD_LEAP){
                        if(jd + SIMD_LEAP <= mat2_block.cols){
                            if(id + 4 <= mat1_block.rows){
                                avx_kernel_4x8(
                                    mat1_block, mat2_block, mat3,
                                    (int)id, (int)jd,
                                    i_start, j_start,
                                    mat1_block.cols
                                );
                            }else{
                                for(size_t tail_id = id; tail_id < mat1_block.rows; tail_id++){
                                    avx_kernel_1x8(
                                        mat1_block, mat2_block, mat3,
                                        (int)tail_id, (int)jd,
                                        i_start, j_start,
                                        mat1_block.cols
                                    );
                                }
                            }
                        }else{
                            size_t id_end = MIN(id + 4, mat1_block.rows);
                            for(size_t tail_id = id; tail_id < id_end; tail_id++){
                                for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                    float sum = 0.0f;
                                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                        sum += *(mat1_block.data + tail_id * mat1_block.cols + kd) *
                                               *(mat2_block.data + kd * mat2_block.cols + tail_jd);
                                    }
                                    *(mat3.data + (i_start + tail_id) * mat3.cols + j_start + tail_jd) += sum;
                                }
                            }
                        }
                    }
                }
            }
            free(mat1_block.data);
            free(mat2_block.data);
        }
    }
}

//加入OpenMP多线程和预取指令的版本
void matmul_v6_omp_avx_6x16(
    struct Matrix mat1, struct Matrix mat2, struct Matrix mat3
){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}//入口检查

    #pragma omp parallel
    {
        struct Matrix mat1_block; struct Matrix mat2_block;
        mat1_block.data = aligned_alloc(64, (size_t) BLOCK_LEAP * BLOCK_LEAP * sizeof(float));
        mat2_block.data = aligned_alloc(64, (size_t) BLOCK_LEAP * BLOCK_LEAP * sizeof(float));

        if(mat1_block.data == NULL || mat2_block.data == NULL){//错误处理
            fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n",BLOCK_LEAP, BLOCK_LEAP);
            free(mat1_block.data);
            free(mat2_block.data);
            exit(-1);
        }

        #pragma omp for collapse(2)
        for(size_t i = 0; i < mat1.rows; i+=BLOCK_LEAP){
            for(size_t j = 0; j < mat2.cols; j+=BLOCK_LEAP){

            for(size_t k = 0; k < mat1.cols; k+=BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);//要考虑边界
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);

                mat1_block.rows = i_end - i_start; mat1_block.cols = k_end - k_start;
                mat2_block.rows = k_end - k_start; mat2_block.cols = j_end - j_start;//无论如何，row/col永远比BLOCK_LEAP小

                //pack mat1,这里虽似乎多此一举，但用一个双重循环换来未来上百次的内存连续访问
                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                        *(mat1_block.data + id*mat1_block.cols + kd) = 
                        *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                    }
                }

                //pack mat2
                for(size_t kd = 0; kd < mat2_block.rows; kd++){
                    for(size_t jd = 0; jd < mat2_block.cols; jd++){
                        *(mat2_block.data + kd * mat2_block.cols + jd) = 
                        *(mat2.data + (k_start + kd) * mat2.cols + j_start + jd);
                    }
                }

                for(size_t id = 0; id < mat1_block.rows;){
                    if(id + 6 <= mat1_block.rows){//宽度能够塞满6行的算子
                        size_t jd = 0;
                        for(; jd  + 16 <= mat2_block.cols; jd += 16){
                            avx_kernel_6x16(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd, 
                                i_start, j_start, 
                                mat1_block.cols);
                        }
                        for(; jd + 8 <= mat2_block.cols; jd += 8){
                            avx_kernel_6x8(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd, 
                                i_start, j_start, 
                                mat1_block.cols);
                        }
                        for(size_t row = 0; row < 6; row++){
                            float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                            for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                float *mat2_row = mat2_block.data + kd * mat2_block.cols;
                                for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                    *(mat3_row + tail_jd) += a * *(mat2_row + tail_jd);
                                }
                            }
                        }
                        id += 6;
                    }
                    else if(id + 4 <= mat1_block.rows){//宽度能否够塞满4行的算子
                        size_t jd = 0;
                        for(; jd  + 16 <= mat2_block.cols; jd += 16){
                            avx_kernel_4x8(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd, 
                                i_start, j_start, 
                                mat1_block.cols
                            );
                            avx_kernel_4x8(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd + 8, 
                                i_start, j_start, 
                                mat1_block.cols);
                        }
                        for(; jd + 8 <= mat2_block.cols; jd += 8){
                            avx_kernel_4x8(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd, 
                                i_start, j_start, 
                                mat1_block.cols
                            );
                        }
                        for(size_t row = 0; row < 4; row++){
                            float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                            for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                float *mat2_row = mat2_block.data + kd * mat2_block.cols;
                                for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                    *(mat3_row + tail_jd) += a * *(mat2_row + tail_jd);
                                }
                            }
                        }
                        id += 4;
                    }
                    else if(id + 2 <= mat1_block.rows){
                        size_t jd = 0;
                        for(; jd + 16 <= mat2_block.cols; jd += 16){
                            avx_kernel_2x16(
                                mat1_block, 
                                mat2_block, 
                                mat3, 
                                id, jd, 
                                i_start, j_start, 
                                mat1_block.cols
                            );
                        }
                        for(; jd + 8 <= mat2_block.cols; jd += 8){
                            avx_kernel_2x8(
                                mat1_block,
                                mat2_block,
                                mat3,
                                id, jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
                        }
                        for(size_t row = 0; row < 2; row++){
                            float *mat3_row = mat3.data + (i_start + id + row) * mat3.cols + j_start;
                            for(size_t kd = 0; kd < mat1_block.cols; kd++){
                                float a = *(mat1_block.data + (id + row) * mat1_block.cols + kd);
                                float *mat2_row = mat2_block.data + kd * mat2_block.cols;
                                for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                    *(mat3_row + tail_jd) += a * *(mat2_row + tail_jd);
                                }
                            }
                        }
                        id += 2;
                    }
                    else{//只能塞下不够1行算子
                        size_t jd = 0;
                        for(; jd + 8 <= mat2_block.cols; jd += 8){
                            avx_kernel_1x8(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
                        }
                        float *mat3_row = mat3.data + (i_start + id) * mat3.cols + j_start;
                        for(size_t kd = 0; kd < mat1_block.cols; kd++){
                            float a = *(mat1_block.data + id * mat1_block.cols + kd);
                            float *mat2_row = mat2_block.data + kd * mat2_block.cols;
                            for(size_t tail_jd = jd; tail_jd < mat2_block.cols; tail_jd++){
                                *(mat3_row + tail_jd) += a * *(mat2_row + tail_jd);
                            }
                        }
                        id++;
                    }
                }
            }
        }
        }
        free(mat1_block.data);
        free(mat2_block.data);
    }
}

void matmul_v7_avx512_omp(
    struct Matrix mat1, struct Matrix mat2, struct Matrix mat3
){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}//入口检查

    #pragma omp parallel
    {
        struct Matrix mat1_block; struct Matrix mat2_block;
        mat1_block.data = aligned_alloc(64, (size_t) BLOCK_LEAP * BLOCK_LEAP * sizeof(float));
        mat2_block.data = aligned_alloc(64, (size_t) BLOCK_LEAP * BLOCK_LEAP * sizeof(float));

        if(mat1_block.data == NULL || mat2_block.data == NULL){
            fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n",BLOCK_LEAP, BLOCK_LEAP);
            free(mat1_block.data);
            free(mat2_block.data);
            exit(-1);
        }

        #pragma omp for collapse(2)
        for(size_t i = 0; i < mat1.rows; i+=BLOCK_LEAP){
            for(size_t j = 0; j < mat2.cols; j+=BLOCK_LEAP){

            for(size_t k = 0; k < mat1.cols; k+=BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);

                mat1_block.rows = i_end - i_start; mat1_block.cols = k_end - k_start;
                mat2_block.rows = k_end - k_start; mat2_block.cols = j_end - j_start;
                
                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                        *(mat1_block.data + id * mat1_block.cols + kd) =
                        *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                    }
                }

                for(size_t kd = 0; kd < mat2_block.rows; kd++){
                    for(size_t jd = 0; jd < mat2_block.cols; jd++){
                        *(mat2_block.data + kd * mat2_block.cols + jd) =
                        *(mat2.data + (k_start + kd) * mat2.cols + j_start + jd);
                    }
                }

                for(size_t id = 0; id < mat1_block.rows;){
                    if(id + 6 <= mat1_block.rows){
                        size_t jd = 0;
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_6x32(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_4x32(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_2x32(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_1x32(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
        free(mat2_block.data);
    }
}

#ifndef II
#define II 132
#endif
#ifndef JJ
#define JJ 8640
#endif
#ifndef KK
#define KK 160
#endif

void matmul_v8_avx512_omp_improved(
    struct Matrix mat1, struct Matrix mat2, struct Matrix mat3
){

    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}//入口检查

    struct Matrix mat2_block;
    size_t mat2_block_panels = ((size_t)JJ + NR - 1) / NR;//向上取整
    //优化访存
    mat2_block.data = aligned_alloc(64, mat2_block_panels * (size_t)KK * NR * sizeof(float));

    if(mat2_block.data == NULL){
        fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n",BLOCK_LEAP, BLOCK_LEAP);
        free(mat2_block.data);
        exit(-1);
    }

    #pragma omp parallel
    {
        struct Matrix mat1_block;
        mat1_block.data = aligned_alloc(64, (size_t) II * KK * sizeof(float));

        if(mat1_block.data == NULL){
            fprintf(stderr, "Error: malloc hit memory limit for block size %d * %d.\n",BLOCK_LEAP, BLOCK_LEAP);
            free(mat1_block.data);
            exit(-1);
        }

        for(size_t j = 0; j < mat2.cols; j+=JJ){
                size_t j_start = j, j_end = MIN(j + JJ, mat2.cols);

            for(size_t k = 0; k < mat1.cols; k+=KK){
                size_t k_start = k, k_end = MIN(k + KK, mat1.cols);

                #pragma omp single
                {
                    mat2_block.rows = k_end - k_start; mat2_block.cols = j_end - j_start;

                    size_t current_panels = ((size_t)mat2_block.cols + NR - 1) / NR;//在现在的参数限制之下，一共有多少panel
                    for(size_t panel = 0; panel < current_panels; panel++){
                        size_t panel_cols = MIN((size_t)NR, mat2_block.cols - panel * NR);
                        for(size_t kd = 0; kd < mat2_block.rows; kd++){
                            float *dst = mat2_block.data + panel * mat2_block.rows * NR + kd * NR;//对应行为：kd + k_start,对应列为panel*32 + jd
                            float *src = mat2.data + (k_start + kd) * mat2.cols + j_start + panel * NR;
                            for(size_t inner = 0; inner < panel_cols; inner++){
                                *(dst + inner) = *(src + inner);//填充32行或者尾行
                            }
                        }
                    }
                }

                #pragma omp for
                for(size_t i = 0; i < mat1.rows; i+=II){
                size_t i_start = i, i_end = MIN(i + II, mat1.rows);

                mat1_block.rows = i_end - i_start; mat1_block.cols = k_end - k_start;

                for(size_t id = 0; id < mat1_block.rows; id++){
                    for(size_t kd = 0; kd < mat1_block.cols; kd++){
                        *(mat1_block.data + id * mat1_block.cols + kd) =
                        *(mat1.data + (i_start + id) * mat1.cols + k_start + kd);
                    }
                }

                for(size_t id = 0; id < mat1_block.rows;){
                    if(id + 6 <= mat1_block.rows){
                        size_t jd = 0;
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_6x32_panelb(//写了一个新的kernel，专门针对这种数据存储结构
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
                        }
                        for(size_t row = 0; row < 6; row++){//做普通乘法
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_4x32_panelb(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_2x32_panelb(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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
                        for(; jd + 32 <= mat2_block.cols; jd += 32){
                            avx512_kernel_1x32_panelb(
                                mat1_block,
                                mat2_block,
                                mat3,
                                (int)id, (int)jd,
                                i_start, j_start,
                                mat1_block.cols
                            );
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

void matmul_v9_OpenBLAS(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}

    //调用cblas_sgemm函数，参数详见OpenBLAS文档
    cblas_sgemm(
    CblasRowMajor,      // 行主序
    CblasNoTrans,       // A 不转置
    CblasNoTrans,       // B 不转置
    (int)mat1.rows,     // M
    (int)mat2.cols,     // N
    (int)mat1.cols,     // K
    1.0f,               // alpha
    mat1.data,          // A
    (int)mat1.cols,     // lda
    mat2.data,          // B
    (int)mat2.cols,     // ldb
    0.0f,               // beta：覆盖 C(C = alpha * A * B + beta * C
    mat3.data,          // C
    (int)mat3.cols      // ldc
    );
}

void multiply_improved(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3){
    if(check_matrix(mat1, mat2, mat3) == -1){exit(-1);}
    matmul_v7_avx512_omp(mat1, mat2, mat3);
}

Transform get_transform(float max_val, float min_val){
    Transform t;
    t.scale = (max_val - min_val) / 255.0f;//每个整数对应float走多少
    t.zero_point = (int)roundf(-128.0f - min_val / t.scale);//0对应的整数值
    return t;
}

MatrixINT8 transfrom_float_to_int8(struct Matrix mat, Transform t){
    MatrixINT8 matint;
    matint.rows = mat.rows;
    matint.cols = mat.cols;
    matint.data = malloc(sizeof(int8_t) * matint.rows * matint.cols);
    if(matint.data == NULL){
        fprintf(stderr, "Error: malloc failed.\n");
        exit(-1);
    }

    for(size_t i = 0; i < mat.rows * mat.cols; i++){
        int q = (int)roundf(*(mat.data + i) / t.scale) + t.zero_point;
        if(q > 127){
            q = 127;
        }
        if(q < -128){
            q = -128;
        }
        *(matint.data + i) = (int8_t)q;
    }
    return matint;
}

int check_matrix_int8(MatrixINT8 mat1, MatrixINT8 mat2, MatrixINT32 mat3){
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
        exit(-1);
    }
    return 0;
}

void matmul_int8(MatrixINT8 mat1, MatrixINT8 mat2, MatrixINT32 mat3, Transform tA, Transform tB){
    if(check_matrix_int8(mat1, mat2, mat3) == -1){
        exit(-1);
    }

    for(size_t i = 0; i < mat3.rows * mat3.cols; i++){
        *(mat3.data + i) = 0;
    }

    //挑选分块矩阵
    for (size_t i = 0; i < mat1.rows; i += BLOCK_LEAP){
        for(size_t j = 0; j < mat2.cols; j += BLOCK_LEAP){
            for(size_t k = 0; k < mat1.cols; k = k + BLOCK_LEAP){
                size_t i_start = i, i_end = MIN(i + BLOCK_LEAP, mat1.rows);//要考虑边界
                size_t j_start = j, j_end = MIN(j + BLOCK_LEAP, mat2.cols);
                size_t k_start = k, k_end = MIN(k + BLOCK_LEAP, mat1.cols);
                //小矩阵内乘法
                for(size_t id = i_start; id < i_end; id++){
                    for(size_t kd = k_start; kd < k_end; kd++){
                        int32_t a = (int32_t)*(mat1.data + id*mat1.cols + kd) - tA.zero_point;
                        for(size_t jd = j_start; jd < j_end; jd++){
                            int32_t b = (int32_t)*(mat2.data + kd*mat2.cols + jd) - tB.zero_point;
                            *(mat3.data + id*mat3.cols + jd) += a * b;
                        }
                    }
                }
            }
        }
    }
}

struct Matrix transform_int32_to_float(MatrixINT32 mat, Transform tA, Transform tB){
    struct Matrix mat_float;
    mat_float.rows = mat.rows;
    mat_float.cols = mat.cols;
    mat_float.data = malloc(sizeof(float) * mat.rows * mat.cols);
    if(mat_float.data == NULL){
        fprintf(stderr, "Error: malloc failed.\n");
        exit(-1);
    }

    for(size_t i = 0; i < mat.rows * mat.cols; i++){
        *(mat_float.data + i) = (float)((double)*(mat.data + i) * (double)tA.scale * (double)tB.scale);
    }
    return mat_float;
}

void compare_error(struct Matrix expected, struct Matrix actual){
    if(expected.data == NULL || actual.data == NULL){
        fprintf(stderr, "Error: matrix data pointer is NULL.\n");
        return;
    }
    if(expected.rows != actual.rows || expected.cols != actual.cols){
        fprintf(stderr, "Error: matrix dimensions do not match for error comparison.\n");
        return;
    }

    double sum_abs_error = 0.0;
    double sum_abs_expected = 0.0;
    double max_abs_error = 0.0;
    size_t max_error_i = 0;
    size_t max_error_j = 0;

    for(size_t i = 0; i < expected.rows; i++){
        for(size_t j = 0; j < expected.cols; j++){
            size_t index = i * expected.cols + j;
            double expected_val = (double)expected.data[index];
            double actual_val = (double)actual.data[index];
            double abs_error = expected_val - actual_val;
            if(abs_error < 0.0){
                abs_error = -abs_error;
            }

            double abs_expected = expected_val;
            if(abs_expected < 0.0){
                abs_expected = -abs_expected;
            }

            sum_abs_error += abs_error;
            sum_abs_expected += abs_expected;
            if(abs_error > max_abs_error){
                max_abs_error = abs_error;
                max_error_i = i;
                max_error_j = j;
            }
        }
    }

    if(sum_abs_expected == 0.0){
        printf("ERROR_COMPARE: sum_abs_error=%e, sum_abs_expected=0, error_ratio=inf, max_abs=%e at (%zu, %zu)\n",
                sum_abs_error,
                max_abs_error,
                max_error_i,
                max_error_j);
        return;
    }

    printf("ERROR_COMPARE: sum_abs_error=%e, sum_abs_expected=%e, error_ratio=%e, max_abs=%e at (%zu, %zu)\n",
            sum_abs_error,
            sum_abs_expected,
            sum_abs_error / sum_abs_expected,
            max_abs_error,
            max_error_i,
            max_error_j);
}
