#include "multiply.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MEASURE_TIME 20

static double time_ms(void){
    struct timespec ts;//结构体，包含tc_sec和tv_nsec，分别测量秒和纳秒
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

//生成-12345 ~ 54321之间的整数
static void set_largenum_mat(struct Matrix mat){
    for(size_t i = 0; i < mat.rows * mat.cols; i++){
        float matmin = -12345.f;
        float matmax = 54321.f;
        float length = matmax - matmin;
        float ratio = (float)rand() / RAND_MAX;
        *(mat.data + i) = matmin + ratio * length;
    }
}

int main(void){
    //INT8 Quantized MatMul
    struct Matrix mat1; struct Matrix mat2; struct Matrix mat3;
    mat1.rows = 3000; mat1.cols = 3000;
    mat2.rows = 3000; mat2.cols = 3000;
    mat3.rows = 3000; mat3.cols = 3000;

    mat1.data = malloc(sizeof(float) * mat1.rows * mat1.cols);
    mat2.data = malloc(sizeof(float) * mat2.rows * mat2.cols);
    mat3.data = malloc(sizeof(float) * mat3.rows * mat3.cols);
    if(mat1.data == NULL || mat2.data == NULL || mat3.data == NULL){
        fprintf(stderr, "Error: malloc failed for INT8 test matrices.\n");
        free(mat1.data);
        free(mat2.data);
        free(mat3.data);
        exit(-1);
    }

    set_largenum_mat(mat1);
    set_largenum_mat(mat2);
    memset(mat3.data, 0, sizeof(float) * mat3.rows * mat3.cols);

    //普通先测试一次
    double total_time = 0.0;
    for(int i = 0; i < MEASURE_TIME; i++){
        memset(mat3.data, 0, sizeof(float) * mat3.rows * mat3.cols);
        double start = time_ms();
        matmul_v3_block(mat1, mat2, mat3);//这里统一用分块乘法的函数
        double end = time_ms();
        total_time += end - start;
    }
    double avg_time_float = total_time / MEASURE_TIME;
    printf("Float Matrix Multiplication AVERAGE_TIME: %f ms\n", avg_time_float);

    //获取transform参数
    Transform tA = get_transform(54321.f, -12345.f);
    Transform tB = get_transform(54321.f, -12345.f);

    MatrixINT8 mat1_int8 = transfrom_float_to_int8(mat1, tA);
    MatrixINT8 mat2_int8 = transfrom_float_to_int8(mat2, tB);

    MatrixINT32 mat3_int32;//int32用来存储结果
    mat3_int32.rows = mat3.rows;
    mat3_int32.cols = mat3.cols;
    mat3_int32.data = malloc(sizeof(int32_t) * mat3_int32.rows * mat3_int32.cols);
    if(mat3_int32.data == NULL){
        fprintf(stderr, "Error: malloc failed for INT32 result matrix.\n");
        free(mat1_int8.data);
        free(mat2_int8.data);
        free(mat3.data);
        free(mat1.data);
        free(mat2.data);
        exit(-1);
    }

    //计算
    total_time = 0.0;
    for(int i = 0; i < MEASURE_TIME; i++){
        double t1 = time_ms();
        matmul_int8(mat1_int8, mat2_int8, mat3_int32, tA, tB);
        double t2 = time_ms();
        total_time += t2 - t1;
    }
    double avg_time_int32 = total_time / MEASURE_TIME;
    printf("INT32 Matrix Multiplication AVERAGE_TIME: %f ms\n", avg_time_int32);

    //输出比较结果
    if(avg_time_float > avg_time_int32){
        double percent = (avg_time_float - avg_time_int32) / avg_time_float * 100.0;
        printf("INT32 is %f%% faster than float\n", percent);//后面两个%%的第一个是转义
    }
    else{
        double percent = (avg_time_int32 - avg_time_float) / avg_time_int32 * 100.0;
        printf("Float is %f%% faster than INT32\n", percent);
    }

    //把结果转换到float
    struct Matrix mat3_trans = transform_int32_to_float(mat3_int32, tA, tB);
    compare_error(mat3, mat3_trans);

    free(mat3_trans.data);
    free(mat3_int32.data);
    free(mat1_int8.data);
    free(mat2_int8.data);
    free(mat3.data);
    free(mat1.data);
    free(mat2.data);
    return 0;
}
