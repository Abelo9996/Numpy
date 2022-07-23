#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in matrix.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* Check for incorrect dimensions, return ValueError if found */
    if ((rows <= 0) || (cols <= 0)){
        PyErr_SetString(PyExc_ValueError, "Dimensions need to be positive!");
        return -1;
    }
    /* Allocate matrix space */
    *mat = (matrix *) malloc(sizeof(matrix));
    /* Return error for a failure */
    if (*mat == NULL){
        PyErr_SetString(PyExc_RuntimeError, "Error upon opening up space!");
        return -1;
    }
    /* Set matrix info */
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    /* Allocate double pointer to matrix */
    (*mat)->data = (double **) malloc(rows*sizeof(double *));
    /* Check for malloc error, free matrix and return RunTimeError if found. */
    if ((*mat)->data == NULL){
        free(*mat);
        PyErr_SetString(PyExc_RuntimeError, "Error upon opening up space!");
        return -1;
    }
    /* Calloc matrix's single pointer with 0s (calloc's purpose) */
    *((*mat)->data) = (double *) calloc(1,rows*cols*sizeof(double));
    /* Check for malloc error, free matrix and data, and return RuntimeError if found. */
    if (*((*mat)->data) == NULL){
        free((*mat)->data);
        free(*mat);
        PyErr_SetString(PyExc_RuntimeError, "Error upon opening up space!");
        return -1;
    }
    /* Set matrix 1d info and return 0 for success. */
    if (rows == 1 || cols == 1){
        (*mat)->is_1d = 1;
    }else{
        (*mat)->is_1d = 0;
    }
    for (int row = 0; row < rows; row++){
        ((*mat)->data)[row] = (*((*mat)->data)+cols*row);
    }
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in matrix.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {

    /* Check for incorrect dimensions, return RunTimeError if found */
    if (rows <= 0 || cols <= 0 || col_offset+cols > from->cols || row_offset+rows>from->rows){
        PyErr_SetString(PyExc_RuntimeError, "Indexing error!");
        return -1;    
    }
    /* Malloc space for matrix */
    *mat = (matrix *) malloc(sizeof(matrix));
    /* Check for matrix error, return RunTimeError if found. */
    if (*mat == NULL){
        PyErr_SetString(PyExc_RuntimeError, "Error upon opening up space!");
        return -1;
    }
    /* Malloc single pointers to data. */
    (*mat)->data = malloc(rows*sizeof(double *));
    /* Set matrix info. */
    from->ref_cnt = from->ref_cnt+1;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = from;
    /* Check for malloc errors, return RuntimeError and free mat if found. */
    if ((*mat)->data == NULL){
        free(*mat);
        PyErr_SetString(PyExc_RuntimeError, "Error upon opening up space!");
        return -1;
    }
    /* Set 1d info */
    if (rows == 1 || cols == 1){
        (*mat)->is_1d = 1;
    }else{
        (*mat)->is_1d = 0;
    }
    /* Set row_offset and col_offset to our matrix and then finally return 0 for success. */
    for (int row = 0; row < rows; row++){
            ((*mat)->data)[row] = (((from->data)[row+row_offset])+col_offset);
    }
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* Don't do anything if matrix is null */
    if (mat == NULL){
        return;
    }
    /* If no parent, reduce reference counter and then free all of matrix data if no other matrices point to mat. */
    if (mat->parent == NULL){
        (mat->ref_cnt)--;
        if (mat->ref_cnt == 0){
            free((mat->data)[0]);
            free(mat->data);
            free(mat);
        }
    /* If there is a parent, reduce reference counter and then free all of parent's data if no other matrices point to mat->parent, then free mat (but not mat->data). */
    }else{
        (mat->parent->ref_cnt)--;
        if (mat->parent->ref_cnt == 0){
            free((mat->parent->data)[0]);
            free(mat->parent->data);
            free(mat->parent);
        }
        free(mat);
    }
    /* Finally, return. */
    return;
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* Get value with the given row and col by indexing into mat->data. */
    return ((mat->data)[row])[col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* Set value with the given row and col by indexing into mat->data. */
    ((mat->data)[row])[col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* Run pragma for optimized performance */
    #pragma omp parallel for
    for (int i = 0; i < ((mat->rows*mat->cols)/4)*4; i+=4){
          /* Access and store each value (aka val) within mat->data */
        _mm256_storeu_pd(((mat->data)[0])+i, _mm256_set1_pd(val));
    }
    for (int i = ((mat->rows*mat->cols)/4)*4; i < mat->rows*mat->cols; i++){
          /* Finally, set last value as val. */
        ((mat->data)[0])[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* Check for dimension errors and return -1 for failure */
    if (result == NULL || mat1 == NULL || mat2 == NULL || mat1->rows != mat2->rows || mat1->cols != mat2->cols || mat1->rows != result->rows || mat1->cols != result->cols){
        return -1;
    }
    /* Run pragma for optimized performance */
    #pragma omp parallel for
    for (int i = 0; i < ((mat2->rows*mat2->cols)/4)*4; i += 4){
        /* Add matrices' values together and store them within result->data*/
        _mm256_storeu_pd(((result->data)[0])+i, _mm256_add_pd(_mm256_loadu_pd(((mat1->data)[0])+i), _mm256_loadu_pd(((mat2->data)[0])+i)));
    }
    for (int i = ((mat1->rows*mat1->cols)/4)*4; i < (mat1->rows*mat1->cols); i++){
        /* Finally, set last value within result->data. */
        ((result->data)[0])[i] = ((mat1->data)[0])[i] + ((mat2->data)[0])[i];
    }
    /* Return 0 for success. */
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* Check for dimension errors and return -1 for failure */
    if (result == NULL || mat1 == NULL || mat2 == NULL || mat1->rows != mat2->rows || mat1->cols != mat2->cols || mat1->rows != result->rows || mat1->cols != result->cols){
        return -1;
    }
    /* Run pragma for optimized performance */
    #pragma omp parallel for
    for (int i = 0; i < ((mat2->rows*mat2->cols)/4)*4; i += 4){
        /* Add matrices' values together and store them within result->data*/
        _mm256_storeu_pd(((result->data)[0])+i, _mm256_sub_pd(_mm256_loadu_pd(((mat1->data)[0])+i), _mm256_loadu_pd(((mat2->data)[0])+i)));
    }
    for (int i = ((mat1->rows*mat1->cols)/4)*4; i < (mat1->rows*mat1->cols); i++){
        /* Finally, set last value within result->data. */
        ((result->data)[0])[i] = ((mat1->data)[0])[i] - ((mat2->data)[0])[i];
    }
    /* Return 0 for success. */
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* Check for incorrect dimensions and return -1 for failure if found. */
    if (result == NULL || mat1 == NULL || mat2 == NULL|| mat1->rows != result->rows || mat2->cols != result->cols || mat1->cols != mat2->rows){
        return -1;
    }
    /* Initialize variables for pragma. */
    __m256d val1,val2,lstone[((mat2->cols)/4)],emptval,lsttwo[((mat2->cols)/4)];
    /* Set iteration integer and case value (personal testing value) */
    int iter,ourcase;
    /* Set value, which is double */
    double val;
    /* Set result to be completely 0. */
    fill_matrix(result, 0);
    /* Run pragma for optimized performance and define private variables */
    #pragma omp parallel for private(lsttwo,lstone,iter, ourcase, val1, val2, emptval, val)
      /* Iterate through and set ourcase to 19 if mat1 rows are not equal to i. */
      for (int i = 0; i <= (mat1->rows/2) * 2; i += 2) {
      iter=2;ourcase = 19;
      if (i!=(mat1->rows/2)*2){} else{if (i!=mat1->rows){iter = 1;ourcase = -19;}else{continue;}}
        /* Set memory for 2 arrays that will be used to access values from. */
        if (ourcase == 19) {
          memset(lstone, 0, mat2->cols/4*sizeof(__m256d));
          memset(lsttwo, 0, mat2->cols/4*sizeof(__m256d));
        }else{
          memset(lstone, 0, mat2->cols/4*sizeof(__m256d));
        }
        /* Iterate through to get the values when multiplying, then set these values within result. */
        for (int k = 0; k<mat1->cols / 4 * 4; k += 4) {
          for (int x = 0; x<iter; x++){
            for (int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
              /* Get values. */
              val = get(result, i + x, j)+get(mat2, k, j) * get(mat1, i+x, k) + get(mat2, k+1, j) * get(mat1, i+x, k+1)+get(mat2, k+2, j) * get(mat1, i+x, k+2)+get(mat2, k+3, j) * get(mat1, i+x, k+3);
              /* Set values. */
              set(result, i + x, j, val);
            }
          }
          /* Run pragma for optimized performance */
          for (int j = 0; j < mat2->cols / 4* 4; j += 4){
            /* Load, set, then add all of the values for mat1 and mat2 together, then store within our first array called lstone. */
            /* First value */
            val1 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + k*mat2->cols), lstone[j/4]);
            /* Second value */
            val1 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+1 + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+1)*mat2->cols), val1);
            /* Third value */
            val1 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+2 + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+2)*mat2->cols), val1);
            /* Fourth value */
            val1 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+3 + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+3)*mat2->cols), val1);
            lstone[j/4] = val1;
            /* If case from above is satisfied, Load, set, then add all of the values for mat1 and mat2 together, then store within our second array called lsttwo. */
            if (ourcase == 19) {
              /* First value */
              val2 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k + (i+1)*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + k*mat2->cols), lsttwo[j/4]);
              /* Second value. */
              val2 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+1 + (i+1)*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+1)*mat2->cols), val2);
              /* Third value. */
              val2 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+2 + (i+1)*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+2)*mat2->cols), val2);
              /* fourth value. */
              val2 = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k+3 + (i+1)*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + (k+3)*mat2->cols), val2);
              lsttwo[j/4] = val2;
            }
          }
        }
        /* Iterate through matrix data */
        for (int k = mat1->cols / 4 * 4; k < mat1->cols; k++) {
          for (int x = 0; x < iter; x++){
            for (int j = mat2->cols / 4 * 4; j < mat2->cols; j++) {
              /* Set results' value by getting mat1's row, multiplying mat1's row with column and then adding to value. */
              set(result, i + x, j, get(result, i+x, j) + get(mat1, i+x, k) * get(mat2, k, j));
            }
          }
          /* Iterate through our stored arrays from above. */
          for (int j = 0; j < mat2->cols / 4* 4; j += 4){
            /* Check if our case from above is satisfied */
            if (ourcase == 19) {
              /* If satisfied, do same process as above and store. */
              lstone[j/4] = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + k*mat2->cols), lstone[j/4]);
              /* If satisfied, do same process as above and store. */
              lsttwo[j/4] = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k + (i+1)*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + k*mat2->cols), lsttwo[j/4]);
            }else{
              /* If not satisfied, do same process as above and store but just for our first array. */
              lstone[j/4] = _mm256_fmadd_pd(_mm256_set1_pd(*((mat1->data[0])+ k + i*mat1->cols)), _mm256_loadu_pd((mat2->data[0])+ j + k*mat2->cols), lstone[j/4]);
            }
          }
        }
        /* Finally, repeat same process from above. */
        for (int j = 0; j < mat2->cols/4 *4; j+= 4 ){
          /* Check if our case is satisfied from above */
          if (ourcase == 19){
            /* If so, do same process from above with our array's values incorporated */
            _mm256_storeu_pd((result->data[0])+ j + i*mat2->cols, lstone[j/4]);
            /* If so, do same process from above with our array's values incorporated */
            _mm256_storeu_pd((result->data[0])+ j + (i+1)*mat2->cols, lsttwo[j/4]);
          }else{
            /* If so, do same process from above with our array's values incorporated (just for the first array). */
            _mm256_storeu_pd((result->data[0])+ j + i*mat2->cols, lstone[j/4]);
          }
        }
      }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* Check for incorrect dimensions and incorrect arguments, return ValueError if found.*/
    if (result == NULL || mat == NULL || pow < 0 || mat->rows != result->rows || mat->cols != result->cols || mat->cols != mat->rows || result->cols != result->rows){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return -1;
    }
    /* Set temporary value for pow to use.*/
    int ourpow = pow;
    /* Set up temporary matrices for multiplication. */
    matrix *mat1,*mat2,*mat3,*mat4;
    mat1=NULL;
    /* Check for allocation errors, if found, return -1 and error from allocate_matrix. */
    if (allocate_matrix(&mat1, result->rows, result->cols) == -1){
      return -1;
    }
    /* Set mat1 full of 0s */
    fill_matrix(mat1, 0);
    /* Use pragma for optimization of speedup. */
    #pragma omp parallel for
    /* Set mat1 to identity matrix*/
    for (int row = 0; row < ((result->rows / 1) * 1); row++) {
      set(mat1, row, row, 1);
    }
    /* Set mat1's last element to 1 to complete identity matrix. */
    for (int row = ((result->rows / 1) * 1); row < result->rows; row++) {
      set(mat1, row, row, 1);
    }
    /* Check that power is not 0. */
    if (ourpow != 0){
      mat2=NULL;
      /* Allocate space for temporary matrix. */
      if (allocate_matrix(&mat2, result->rows, result->cols) == -1){
        return -1;
      }
      /* Use pragma for speedup optimization. */
      #pragma omp parallel for
      /* Iterate through our given mat */
      for (int i = 0; i < ((mat->rows*mat->cols / 4) * 4); i += 4){
        /* Set our partial mat2 with all values from mat. */
        _mm256_storeu_pd(((mat2->data)[0]) + i, _mm256_loadu_pd(((mat->data)[0]) + i));
      }
      /* Iterate to last element. */
      for (int i = ((mat->rows*mat->cols / 4) * 4); i < mat->rows*mat->cols; i++){
        /* Set last element to the mat value. */
        ((mat2->data)[0])[i] = ((mat->data)[0])[i];
      }
      mat3=mat4= NULL;
      /* Allocate matrix for last matrix needed for multiplication. */
      if (allocate_matrix(&mat3, result->rows, result->cols) == -1){
        return -1;
      }
      while (ourpow != 0){
        if ((ourpow%2) != 1){
        }else {
          /* Multiply mat1 with mat2 if pow is odd. */
          mul_matrix(mat3, mat1, mat2);
          /* Change temporary matrices for next multiplication. */
          mat4=mat1,mat1=mat3,mat3=mat4;
        }
        if ((ourpow >> 1) == 0){
        /* Logically shift 1 by right for next power value. */
        ourpow = ourpow >> 1;
        }else {
          /* Logically shift 1 by right for next power value. */
          ourpow = ourpow >> 1;
          /* Multiply mat2 with mat2. */
          mul_matrix(mat3, mat2, mat2);
          /* Change temporary matrices for next multiplication. */
          mat4=mat2,mat2=mat3,mat3=mat4;
        }
      }
    }
    /* Set result to mat1 and return 0 for success. */
    result->data = mat1->data;
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* Check for incorrect dimensions and arguments, return ValueError if found */
    if (result == NULL || mat == NULL || mat->rows != result->rows || mat->cols != result->cols){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return -1;
    }
    /* Use pragma for optimization of speedup */
    #pragma omp parallel for
    /* Iterate through our given mat. */
    for (int i = 0; i<((mat->rows*mat->cols)/4)*4;i+=4){
          /* Multiply each loaded value by our defined scalar of -1, and then store it. */
        _mm256_storeu_pd(((result->data)[0])+i, _mm256_mul_pd(_mm256_loadu_pd(((mat->data)[0])+i), _mm256_set1_pd(-1)));
    }
    /* Iterate for last element */
    for (int i = ((mat->rows*mat->cols)/4)*4; i < (mat->rows*mat->cols); i++){
          /* Multiply element by negative one and store in result. */
        ((result->data)[0])[i] = ((mat->data)[0])[i]*-1;
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* Check for matrix dimension errors and incorret arguments, return ValueError if found. */
    if (result == NULL || mat == NULL || mat->rows != result->rows || mat->cols != result->cols){
        PyErr_SetString(PyExc_ValueError,"Incorrect Dimensions!");
        return -1;
    }
    /* Use pragma for optimization of speedup */
    #pragma omp parallel for
    /* Iterate through our given mat */
    for (int i = 0; i < ((mat->rows*mat->cols)/4)*4; i += 4){
        /* Use the max operator to check which loaded value of 0-x and x would be larger, then store value in result. */
        _mm256_storeu_pd(((result->data)[0])+i,_mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(0),_mm256_loadu_pd(((mat->data)[0])+i)),_mm256_loadu_pd(((mat->data)[0])+i)));
    }
    /* Iterate through for our last element. */
    for (int i = ((mat->rows*mat->cols)/4)*4; i < (mat->rows*mat->cols); i++){
        /* Use conditional to store positive value if positive, and positive value if negative. 
        I used conditionals as they do not reduce performance as drastically as if statements within calls. */
        ((result->data)[0])[i] = (((mat->data)[0])[i]>0) ? ((mat->data)[0])[i]: -1*((mat->data)[0])[i];
    }
    return 0;
}
