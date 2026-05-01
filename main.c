#include <stdint.h>

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "multiply.h"

#define MEASURE_TIME 5
#define MAX_MEMORY 8ULL*1024*1024*1024
#define MATRIX_ALIGNMENT 32

void *aligned_malloc_safe(size_t bytes){
    size_t aligned_bytes = ((bytes + MATRIX_ALIGNMENT - 1) / MATRIX_ALIGNMENT) * MATRIX_ALIGNMENT;
    return aligned_alloc(MATRIX_ALIGNMENT, aligned_bytes);
}

double time_ms(void){
    struct timespec ts;//结构体，包含tc_sec和tv_nsec，分别测量秒和纳秒
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
void set_mat(struct Matrix mat){
    for(size_t i = 0; i < mat.rows*mat.cols; i++){
        //FPU的硬件结构决定了0-1之间的乘法和大浮点数乘法时间完全相同，为方便我们取0-1
        mat.data[i] = (float)rand() / RAND_MAX;
    }
}

//生成-12345 ~ 54321之间的整数
void set_largenum_mat(struct Matrix mat){
    for(size_t i = 0; i < mat.rows * mat.cols; i++){
        float matmin = -12345.f;
        float matmax = 54321.f;
        float length = matmax - matmin;
        float ratio = (float)rand() / RAND_MAX;
        *(mat.data + i) = matmin + ratio * length;
    }
}

void warmup_multiply(size_t dim, int times){
    struct Matrix mat1, mat2, mat3;

    mat1.rows = dim; mat1.cols = dim;
    mat1.data = aligned_malloc_safe(mat1.rows * mat1.cols * sizeof(float));
    mat2.rows = dim; mat2.cols = dim;
    mat2.data = aligned_malloc_safe(mat2.rows * mat2.cols * sizeof(float));
    mat3.rows = dim; mat3.cols = dim;
    mat3.data = aligned_malloc_safe(mat3.rows * mat3.cols * sizeof(float));

    if(mat1.data == NULL || mat2.data == NULL || mat3.data == NULL){
        fprintf(stderr, "Error: malloc failed for warmup matrices.\n");
        free(mat1.data);
        free(mat2.data);
        free(mat3.data);
        exit(-1);
    }

    set_mat(mat1);
    set_mat(mat2);

    for(int i = 0; i < times; i++){
        memset(mat3.data, 0, sizeof(float) * mat3.rows * mat3.cols);
        matmul_v8_avx512_omp_improved(mat1, mat2, mat3);
    }

    free(mat1.data);
    free(mat2.data);
    free(mat3.data);
}

int main(){
    size_t dim[] = {16000};//测试不同规模的矩阵乘法性能

    //设定最大的内存限制，并确保该内存可以被分配
    if(set_memory_limit((size_t)MAX_MEMORY) != 0){//设置内存限制，防止malloc导致内存爆炸
        fprintf(stderr,"Error: failed to set memory limit.\n");
        exit(-1);
    }
    
    warmup_multiply(1200, 300);

    for(size_t i = 0; i < sizeof(dim) / sizeof(dim[0]); i++){
        struct Matrix mat1, mat2, mat3;
        
        mat1.rows = dim[i]; mat1.cols = dim[i];
        //申请到一块对齐32字节的内存，目的是让loadu的时候对齐cacheline，避免一块连续内存多次读取
        //align表示对齐，safe表示把size补齐到32的倍数，在进行aligned_alloc
        mat1.data = aligned_malloc_safe(mat1.rows*mat1.cols*sizeof(float));
        mat2.rows = dim[i]; mat2.cols = dim[i];
        mat2.data = aligned_malloc_safe(mat2.rows*mat2.cols*sizeof(float));
        mat3.rows = dim[i]; mat3.cols = dim[i];
        mat3.data = aligned_malloc_safe(mat3.rows*mat3.cols*sizeof(float));

        if(mat1.data == NULL || mat2.data == NULL || mat3.data == NULL){
            fprintf(stderr, "Error: malloc hit memory limit for matrix size %zu * %zu.\n", dim[i], dim[i]);
            free(mat1.data);
            free(mat2.data);
            free(mat3.data);
            continue;//因为有可能下一个矩阵大小就不会爆炸了
        }
        
        //初始化矩阵数据
        set_mat(mat1);
        set_mat(mat2);
        for(size_t i = 0; i < mat3.rows*mat3.cols; i++) *(mat3.data + i) = 0.0f;

        double total_time = 0.0;
        double avg_time = 0.0;

        //使用multiply_improved函数进行矩阵乘法，并测量平均时间
        for(int i = 0; i < MEASURE_TIME; i++){
            double start = 0; double end = 0;

            memset(mat3.data, 0, sizeof(float) * mat3.rows * mat3.cols);
            start = time_ms();
            matmul_v8_avx512_omp_improved(mat1, mat2, mat3);
            end = time_ms(); 

            total_time += end - start;
        }
        avg_time = total_time / MEASURE_TIME;
        printf("MATRIX SIZE %zu * %zu\n", mat1.rows, mat2.cols);
        printf("MATMUL: AVERAGE_TIME: %f ms\n", avg_time);

        free(mat3.data);
        free(mat1.data);
        free(mat2.data);
    }
}   
