#include "multiply.h"
#include <immintrin.h>
#include <stddef.h>

/*数学原理： mat1[i][k]和mat2的第k行相乘，累加到mat3的第i行*/
void avx_kernel_1x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int i,//矩阵块内部坐标
    int j,
    size_t i_start,//矩阵左上方坐标
    size_t j_start,
    size_t valid_k)
{
    //1.加载C[i][start + 0-7]当中的内存
    float *mat3_pos = mat3.data + (i + (int)i_start) * mat3.cols + j_start + j;
    __m256 vector_mat3 = _mm256_loadu_ps(mat3_pos);
    for(size_t p = 0; p < valid_k; p++){
        //2.加载八个相同的mat1[i][p]到寄存器
        __m256 vector_mat1 = _mm256_set1_ps(*(mat1.data + i*mat1.cols + p));

        //3.加载八个连续的mat2[p][0-7 + start]
        __m256 vector_mat2 = _mm256_loadu_ps(mat2.data + j + p*mat2.cols);

        //4.累加
        vector_mat3 = _mm256_fmadd_ps(vector_mat1, vector_mat2, vector_mat3);
    }
    //5.写回内存
    _mm256_storeu_ps(mat3_pos, vector_mat3);
}

void avx_kernel_2x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    //加载mat3的行
    float * mat3_pos0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + (size_t)jd + j_start;
    float * mat3_pos1 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + (size_t)jd + j_start;

    //两个累加器
    __m256 acc0 = _mm256_loadu_ps(mat3_pos0);
    __m256 acc1 = _mm256_loadu_ps(mat3_pos1);

    //一个对a的广播寄存器
    __m256 a;
    for(size_t k = 0; k < valid_k; k++){
        __m256 vector_mat2 = _mm256_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd);//b的一个寄存器

        //c = a * b + c(原来的c，也就是acc0-acc3)
        a = _mm256_set1_ps(*(mat1.data + (id + 0) * mat1.cols + k));
        acc0 = _mm256_fmadd_ps(a, vector_mat2, acc0);

        a = _mm256_set1_ps(*(mat1.data + (id + 1) * mat1.cols + k));
        acc1 = _mm256_fmadd_ps(a, vector_mat2, acc1);
    }
    //把acc0（8个连续的元素）写回c当中的内存
    _mm256_storeu_ps(mat3_pos0, acc0);
    _mm256_storeu_ps(mat3_pos1, acc1);
}

void avx_kernel_4x8(//pack_a的前四行n列和pack_b的前四行八列生成matc的四行八列
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,//分块矩阵中的参数
    int jd,
    size_t i_start,//起始的行列参数
    size_t j_start,
    size_t valid_k)
{
    //把mat3的四行8列加载出来
    float *mat3_pos0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos1 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;

    //四个累加器
    __m256 acc0 = _mm256_loadu_ps(mat3_pos0);//c的四行八列
    __m256 acc1 = _mm256_loadu_ps(mat3_pos1);
    __m256 acc2 = _mm256_loadu_ps(mat3_pos2);
    __m256 acc3 = _mm256_loadu_ps(mat3_pos3);

    //一个对a的广播寄存器
    __m256 a;
    for(size_t kd = 0; kd < valid_k; kd++){//pack_a的四行k列
        __m256 vector_mat2 = _mm256_loadu_ps(mat2.data + kd * mat2.cols + (size_t)jd);//b的一个寄存器

        //c = a * b + c(原来的c，也就是acc0-acc3)
        a = _mm256_set1_ps(*(mat1.data + (id + 0) * mat1.cols + kd));
        acc0 = _mm256_fmadd_ps(a, vector_mat2, acc0);

        a = _mm256_set1_ps(*(mat1.data + (id + 1) * mat1.cols + kd));
        acc1 = _mm256_fmadd_ps(a, vector_mat2, acc1);

        a = _mm256_set1_ps(*(mat1.data + (id + 2) * mat1.cols + kd));
        acc2 = _mm256_fmadd_ps(a, vector_mat2, acc2);

        a = _mm256_set1_ps(*(mat1.data + (id + 3) * mat1.cols + kd));
        acc3 = _mm256_fmadd_ps(a, vector_mat2, acc3);
    }
    //把acc0（8个连续的元素）写回c当中的内存
    _mm256_storeu_ps(mat3_pos0, acc0);
    _mm256_storeu_ps(mat3_pos1, acc1);
    _mm256_storeu_ps(mat3_pos2, acc2);
    _mm256_storeu_ps(mat3_pos3, acc3);
}

void avx_kernel_6x8(//pack_a的前六行n列和pack_b的前六行八列生成matc的六行八列
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,//分块矩阵中的参数
    int jd,
    size_t i_start,//起始的行列参数
    size_t j_start,
    size_t valid_k)
{
    //把mat3的六行8列加载出来
    float *mat3_pos0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos1 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 4) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3.data + (i_start + (size_t)id + 5) * mat3.cols + j_start + (size_t)jd;

    //六个累加器
    __m256 acc0 = _mm256_loadu_ps(mat3_pos0);//c的六行八列
    __m256 acc1 = _mm256_loadu_ps(mat3_pos1);
    __m256 acc2 = _mm256_loadu_ps(mat3_pos2);
    __m256 acc3 = _mm256_loadu_ps(mat3_pos3);
    __m256 acc4 = _mm256_loadu_ps(mat3_pos4);
    __m256 acc5 = _mm256_loadu_ps(mat3_pos5);

    //一个对a的广播寄存器
    __m256 a;
    for(size_t kd = 0; kd < valid_k; kd++){//pack_a的六行k列
        __m256 vector_mat2 = _mm256_loadu_ps(mat2.data + kd * mat2.cols + (size_t)jd);//b的一个寄存器

        //c = a * b + c(原来的c，也就是acc0-acc5)
        a = _mm256_set1_ps(*(mat1.data + (id + 0) * mat1.cols + kd));
        acc0 = _mm256_fmadd_ps(a, vector_mat2, acc0);

        a = _mm256_set1_ps(*(mat1.data + (id + 1) * mat1.cols + kd));
        acc1 = _mm256_fmadd_ps(a, vector_mat2, acc1);

        a = _mm256_set1_ps(*(mat1.data + (id + 2) * mat1.cols + kd));
        acc2 = _mm256_fmadd_ps(a, vector_mat2, acc2);

        a = _mm256_set1_ps(*(mat1.data + (id + 3) * mat1.cols + kd));
        acc3 = _mm256_fmadd_ps(a, vector_mat2, acc3);

        a = _mm256_set1_ps(*(mat1.data + (id + 4) * mat1.cols + kd));
        acc4 = _mm256_fmadd_ps(a, vector_mat2, acc4);

        a = _mm256_set1_ps(*(mat1.data + (id + 5) * mat1.cols + kd));
        acc5 = _mm256_fmadd_ps(a, vector_mat2, acc5);
    }
    //把acc0（8个连续的元素）写回c当中的内存
    _mm256_storeu_ps(mat3_pos0, acc0);
    _mm256_storeu_ps(mat3_pos1, acc1);
    _mm256_storeu_ps(mat3_pos2, acc2);
    _mm256_storeu_ps(mat3_pos3, acc3);
    _mm256_storeu_ps(mat3_pos4, acc4);
    _mm256_storeu_ps(mat3_pos5, acc5);
}

void avx_kernel_6x16(//pack_a的前四行n列和pack_b的前四行八列生成matc的四行八列
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,//分块矩阵中的参数
    int jd,
    size_t i_start,//起始的行列参数
    size_t j_start,
    size_t valid_k)
{
    //加载c的6*(16 / 8) = 12个地址
    float* mat3_pos0 = mat3.data + (i_start + id) * mat3.cols + jd + j_start; float* mat3_pos1 = mat3.data + (i_start + id) * mat3.cols + jd + j_start + 8;
    float* mat3_pos2 = mat3.data + (i_start + id + 1) * mat3.cols + jd + j_start; float* mat3_pos3 = mat3.data + (i_start + id + 1) * mat3.cols + jd + j_start + 8;
    float* mat3_pos4 = mat3.data + (i_start + id + 2) * mat3.cols + jd + j_start; float* mat3_pos5 = mat3.data + (i_start + id + 2) * mat3.cols + jd + j_start + 8;
    float* mat3_pos6 = mat3.data + (i_start + id + 3) * mat3.cols + jd + j_start; float* mat3_pos7 = mat3.data + (i_start + id + 3) * mat3.cols + jd + j_start + 8;
    float* mat3_pos8 = mat3.data + (i_start + id + 4) * mat3.cols + jd + j_start; float* mat3_pos9 = mat3.data + (i_start + id + 4) * mat3.cols + jd + j_start + 8;
    float* mat3_pos10 = mat3.data + (i_start + id + 5) * mat3.cols + jd + j_start; float* mat3_pos11 = mat3.data + (i_start + id + 5) * mat3.cols + jd + j_start + 8;

    //12个累加器
    __m256 acc0 = _mm256_loadu_ps(mat3_pos0); __m256 acc1 = _mm256_loadu_ps(mat3_pos1);
    __m256 acc2 = _mm256_loadu_ps(mat3_pos2); __m256 acc3 = _mm256_loadu_ps(mat3_pos3);
    __m256 acc4 = _mm256_loadu_ps(mat3_pos4); __m256 acc5 = _mm256_loadu_ps(mat3_pos5);
    __m256 acc6 = _mm256_loadu_ps(mat3_pos6); __m256 acc7 = _mm256_loadu_ps(mat3_pos7);
    __m256 acc8 = _mm256_loadu_ps(mat3_pos8); __m256 acc9 = _mm256_loadu_ps(mat3_pos9);
    __m256 acc10 = _mm256_loadu_ps(mat3_pos10); __m256 acc11 = _mm256_loadu_ps(mat3_pos11);
    
    //一个对a的广播寄存器
    __m256 a;
    for(size_t k = 0; k < valid_k; k++){
        //两个b内容的寄存器
        __m256 vector1_mat2 = _mm256_loadu_ps(mat2.data + k * mat2.cols + jd);
        __m256 vector2_mat2 = _mm256_loadu_ps(mat2.data + k * mat2.cols + jd + 8);

        a = _mm256_set1_ps(*(mat1.data + (id + 0) * mat1.cols + k));
        acc0 = _mm256_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm256_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm256_set1_ps(*(mat1.data + (id + 1) * mat1.cols + k));
        acc2 = _mm256_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm256_fmadd_ps(a, vector2_mat2, acc3);

        a = _mm256_set1_ps(*(mat1.data + (id + 2) * mat1.cols + k));
        acc4 = _mm256_fmadd_ps(a, vector1_mat2, acc4); 
        acc5 = _mm256_fmadd_ps(a, vector2_mat2, acc5);

        a = _mm256_set1_ps(*(mat1.data + (id + 3) * mat1.cols + k));
        acc6 = _mm256_fmadd_ps(a, vector1_mat2, acc6);
        acc7 = _mm256_fmadd_ps(a, vector2_mat2, acc7);

        a = _mm256_set1_ps(*(mat1.data + (id + 4) * mat1.cols + k));
        acc8 = _mm256_fmadd_ps(a, vector1_mat2, acc8);
        acc9 = _mm256_fmadd_ps(a, vector2_mat2, acc9);

        a = _mm256_set1_ps(*(mat1.data + (id + 5) * mat1.cols + k));
        acc10 = _mm256_fmadd_ps(a, vector1_mat2, acc10);
        acc11 = _mm256_fmadd_ps(a, vector2_mat2, acc11);
    }
    //写回内存
    _mm256_storeu_ps(mat3_pos0, acc0); _mm256_storeu_ps(mat3_pos1, acc1);
    _mm256_storeu_ps(mat3_pos2, acc2); _mm256_storeu_ps(mat3_pos3, acc3);
    _mm256_storeu_ps(mat3_pos4, acc4); _mm256_storeu_ps(mat3_pos5, acc5);
    _mm256_storeu_ps(mat3_pos6, acc6); _mm256_storeu_ps(mat3_pos7, acc7);
    _mm256_storeu_ps(mat3_pos8, acc8); _mm256_storeu_ps(mat3_pos9, acc9);
    _mm256_storeu_ps(mat3_pos10, acc10); _mm256_storeu_ps(mat3_pos11, acc11);
}

void avx_kernel_2x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    //加载c的4个地址
    float* mat3_pos0 = mat3.data + (i_start + id) * mat3.cols + jd + j_start; float* mat3_pos1 = mat3.data + (i_start + id) * mat3.cols + jd + j_start + 8;
    float* mat3_pos2 = mat3.data + (i_start + id + 1) * mat3.cols + jd + j_start; float* mat3_pos3 = mat3.data + (i_start + id + 1) * mat3.cols + jd + j_start + 8;
    
    //4个累加器
    __m256 acc0 = _mm256_loadu_ps(mat3_pos0); __m256 acc1 = _mm256_loadu_ps(mat3_pos1);
    __m256 acc2 = _mm256_loadu_ps(mat3_pos2); __m256 acc3 = _mm256_loadu_ps(mat3_pos3);
    
    //一个对a的广播寄存器
    __m256 a;
    for(size_t k = 0; k < valid_k; k++){
        //两个b内容的寄存器
        __m256 vector1_mat2 = _mm256_loadu_ps(mat2.data + k * mat2.cols + jd);
        __m256 vector2_mat2 = _mm256_loadu_ps(mat2.data + k * mat2.cols + jd + 8);

        a = _mm256_set1_ps(*(mat1.data + (id + 0) * mat1.cols + k));
        acc0 = _mm256_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm256_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm256_set1_ps(*(mat1.data + (id + 1) * mat1.cols + k));
        acc2 = _mm256_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm256_fmadd_ps(a, vector2_mat2, acc3);

        
    }
    //写回内存
    _mm256_storeu_ps(mat3_pos0, acc0); _mm256_storeu_ps(mat3_pos1, acc1);
    _mm256_storeu_ps(mat3_pos2, acc2); _mm256_storeu_ps(mat3_pos3, acc3);
}

#if defined(__GNUC__) || defined(__clang__) //如果使用的gcc或者clang编译器
#define AVX512_TARGET __attribute__((target("avx512f"))) //则告诉编译器，这个函数用avx512编译
#else
#define AVX512_TARGET 
#endif//如果不支持avx512指令，就将其设置成空

AVX512_TARGET
void avx512_kernel_1x32(
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
    float *mat3_pos1 = mat3_pos0 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);

    for(size_t k = 0; k < valid_k; k++){
        __m512 vector1_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd);
        __m512 vector2_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd + 16);
        __m512 a = _mm512_set1_ps(*(mat1.data + (size_t)id * mat1.cols + k));

        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
}

AVX512_TARGET
void avx512_kernel_2x32(
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
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);

    for(size_t k = 0; k < valid_k; k++){
        __m512 vector1_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd);
        __m512 vector2_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
}

AVX512_TARGET
void avx512_kernel_4x32(
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
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3_pos4 + 16;
    float *mat3_pos6 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos7 = mat3_pos6 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);
    __m512 acc4 = _mm512_loadu_ps(mat3_pos4);
    __m512 acc5 = _mm512_loadu_ps(mat3_pos5);
    __m512 acc6 = _mm512_loadu_ps(mat3_pos6);
    __m512 acc7 = _mm512_loadu_ps(mat3_pos7);

    for(size_t k = 0; k < valid_k; k++){
        __m512 vector1_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd);
        __m512 vector2_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 2) * mat1.cols + k));
        acc4 = _mm512_fmadd_ps(a, vector1_mat2, acc4);
        acc5 = _mm512_fmadd_ps(a, vector2_mat2, acc5);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 3) * mat1.cols + k));
        acc6 = _mm512_fmadd_ps(a, vector1_mat2, acc6);
        acc7 = _mm512_fmadd_ps(a, vector2_mat2, acc7);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
    _mm512_storeu_ps(mat3_pos4, acc4);
    _mm512_storeu_ps(mat3_pos5, acc5);
    _mm512_storeu_ps(mat3_pos6, acc6);
    _mm512_storeu_ps(mat3_pos7, acc7);
}

AVX512_TARGET
void avx512_kernel_6x32(
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
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3_pos4 + 16;
    float *mat3_pos6 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos7 = mat3_pos6 + 16;
    float *mat3_pos8 = mat3.data + (i_start + (size_t)id + 4) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos9 = mat3_pos8 + 16;
    float *mat3_pos10 = mat3.data + (i_start + (size_t)id + 5) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos11 = mat3_pos10 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);
    __m512 acc4 = _mm512_loadu_ps(mat3_pos4);
    __m512 acc5 = _mm512_loadu_ps(mat3_pos5);
    __m512 acc6 = _mm512_loadu_ps(mat3_pos6);
    __m512 acc7 = _mm512_loadu_ps(mat3_pos7);
    __m512 acc8 = _mm512_loadu_ps(mat3_pos8);
    __m512 acc9 = _mm512_loadu_ps(mat3_pos9);
    __m512 acc10 = _mm512_loadu_ps(mat3_pos10);
    __m512 acc11 = _mm512_loadu_ps(mat3_pos11);

    for(size_t k = 0; k < valid_k; k++){
        __m512 vector1_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd);
        __m512 vector2_mat2 = _mm512_loadu_ps(mat2.data + k * mat2.cols + (size_t)jd + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 2) * mat1.cols + k));
        acc4 = _mm512_fmadd_ps(a, vector1_mat2, acc4);
        acc5 = _mm512_fmadd_ps(a, vector2_mat2, acc5);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 3) * mat1.cols + k));
        acc6 = _mm512_fmadd_ps(a, vector1_mat2, acc6);
        acc7 = _mm512_fmadd_ps(a, vector2_mat2, acc7);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 4) * mat1.cols + k));
        acc8 = _mm512_fmadd_ps(a, vector1_mat2, acc8);
        acc9 = _mm512_fmadd_ps(a, vector2_mat2, acc9);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 5) * mat1.cols + k));
        acc10 = _mm512_fmadd_ps(a, vector1_mat2, acc10);
        acc11 = _mm512_fmadd_ps(a, vector2_mat2, acc11);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
    _mm512_storeu_ps(mat3_pos4, acc4);
    _mm512_storeu_ps(mat3_pos5, acc5);
    _mm512_storeu_ps(mat3_pos6, acc6);
    _mm512_storeu_ps(mat3_pos7, acc7);
    _mm512_storeu_ps(mat3_pos8, acc8);
    _mm512_storeu_ps(mat3_pos9, acc9);
    _mm512_storeu_ps(mat3_pos10, acc10);
    _mm512_storeu_ps(mat3_pos11, acc11);
}

AVX512_TARGET
void avx512_kernel_1x32_panelb(
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
    float *mat3_pos1 = mat3_pos0 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        __m512 vector1_mat2 = _mm512_loadu_ps(b_ptr);
        __m512 vector2_mat2 = _mm512_loadu_ps(b_ptr + 16);
        __m512 a = _mm512_set1_ps(*(mat1.data + (size_t)id * mat1.cols + k));

        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
}

AVX512_TARGET
void avx512_kernel_2x32_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k
){
    //加载2*16的m3的地址的第一个
    float *mat3_pos0 = mat3.data + (i_start + (size_t)id + 0) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;

    //累加器，一共占有4个avx512寄存器
    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);

    //当前在多少panel，向下取整
    size_t panel = (size_t)jd / NR;
    //找到第panel个32列块
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        __m512 vector1_mat2 = _mm512_loadu_ps(b_ptr);
        __m512 vector2_mat2 = _mm512_loadu_ps(b_ptr + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
}

AVX512_TARGET
void avx512_kernel_4x32_panelb(
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
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3_pos4 + 16;
    float *mat3_pos6 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos7 = mat3_pos6 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);
    __m512 acc4 = _mm512_loadu_ps(mat3_pos4);
    __m512 acc5 = _mm512_loadu_ps(mat3_pos5);
    __m512 acc6 = _mm512_loadu_ps(mat3_pos6);
    __m512 acc7 = _mm512_loadu_ps(mat3_pos7);

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        __m512 vector1_mat2 = _mm512_loadu_ps(b_ptr);
        __m512 vector2_mat2 = _mm512_loadu_ps(b_ptr + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 2) * mat1.cols + k));
        acc4 = _mm512_fmadd_ps(a, vector1_mat2, acc4);
        acc5 = _mm512_fmadd_ps(a, vector2_mat2, acc5);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 3) * mat1.cols + k));
        acc6 = _mm512_fmadd_ps(a, vector1_mat2, acc6);
        acc7 = _mm512_fmadd_ps(a, vector2_mat2, acc7);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
    _mm512_storeu_ps(mat3_pos4, acc4);
    _mm512_storeu_ps(mat3_pos5, acc5);
    _mm512_storeu_ps(mat3_pos6, acc6);
    _mm512_storeu_ps(mat3_pos7, acc7);
}

AVX512_TARGET
void avx512_kernel_6x32_panelb(
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
    float *mat3_pos1 = mat3_pos0 + 16;
    float *mat3_pos2 = mat3.data + (i_start + (size_t)id + 1) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos3 = mat3_pos2 + 16;
    float *mat3_pos4 = mat3.data + (i_start + (size_t)id + 2) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos5 = mat3_pos4 + 16;
    float *mat3_pos6 = mat3.data + (i_start + (size_t)id + 3) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos7 = mat3_pos6 + 16;
    float *mat3_pos8 = mat3.data + (i_start + (size_t)id + 4) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos9 = mat3_pos8 + 16;
    float *mat3_pos10 = mat3.data + (i_start + (size_t)id + 5) * mat3.cols + j_start + (size_t)jd;
    float *mat3_pos11 = mat3_pos10 + 16;

    __m512 acc0 = _mm512_loadu_ps(mat3_pos0);
    __m512 acc1 = _mm512_loadu_ps(mat3_pos1);
    __m512 acc2 = _mm512_loadu_ps(mat3_pos2);
    __m512 acc3 = _mm512_loadu_ps(mat3_pos3);
    __m512 acc4 = _mm512_loadu_ps(mat3_pos4);
    __m512 acc5 = _mm512_loadu_ps(mat3_pos5);
    __m512 acc6 = _mm512_loadu_ps(mat3_pos6);
    __m512 acc7 = _mm512_loadu_ps(mat3_pos7);
    __m512 acc8 = _mm512_loadu_ps(mat3_pos8);
    __m512 acc9 = _mm512_loadu_ps(mat3_pos9);
    __m512 acc10 = _mm512_loadu_ps(mat3_pos10);
    __m512 acc11 = _mm512_loadu_ps(mat3_pos11);

    size_t panel = (size_t)jd / NR;
    float *panel_ptr = mat2.data + panel * valid_k * NR;
    for(size_t k = 0; k < valid_k; k++){
        float *b_ptr = panel_ptr + k * NR;
        __m512 vector1_mat2 = _mm512_loadu_ps(b_ptr);
        __m512 vector2_mat2 = _mm512_loadu_ps(b_ptr + 16);

        __m512 a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 0) * mat1.cols + k));
        acc0 = _mm512_fmadd_ps(a, vector1_mat2, acc0);
        acc1 = _mm512_fmadd_ps(a, vector2_mat2, acc1);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 1) * mat1.cols + k));
        acc2 = _mm512_fmadd_ps(a, vector1_mat2, acc2);
        acc3 = _mm512_fmadd_ps(a, vector2_mat2, acc3);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 2) * mat1.cols + k));
        acc4 = _mm512_fmadd_ps(a, vector1_mat2, acc4);
        acc5 = _mm512_fmadd_ps(a, vector2_mat2, acc5);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 3) * mat1.cols + k));
        acc6 = _mm512_fmadd_ps(a, vector1_mat2, acc6);
        acc7 = _mm512_fmadd_ps(a, vector2_mat2, acc7);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 4) * mat1.cols + k));
        acc8 = _mm512_fmadd_ps(a, vector1_mat2, acc8);
        acc9 = _mm512_fmadd_ps(a, vector2_mat2, acc9);

        a = _mm512_set1_ps(*(mat1.data + ((size_t)id + 5) * mat1.cols + k));
        acc10 = _mm512_fmadd_ps(a, vector1_mat2, acc10);
        acc11 = _mm512_fmadd_ps(a, vector2_mat2, acc11);
    }

    _mm512_storeu_ps(mat3_pos0, acc0);
    _mm512_storeu_ps(mat3_pos1, acc1);
    _mm512_storeu_ps(mat3_pos2, acc2);
    _mm512_storeu_ps(mat3_pos3, acc3);
    _mm512_storeu_ps(mat3_pos4, acc4);
    _mm512_storeu_ps(mat3_pos5, acc5);
    _mm512_storeu_ps(mat3_pos6, acc6);
    _mm512_storeu_ps(mat3_pos7, acc7);
    _mm512_storeu_ps(mat3_pos8, acc8);
    _mm512_storeu_ps(mat3_pos9, acc9);
    _mm512_storeu_ps(mat3_pos10, acc10);
    _mm512_storeu_ps(mat3_pos11, acc11);
}
