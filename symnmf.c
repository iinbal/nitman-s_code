#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define BETA 0.5
#define EPSILON 1e-4
#define SMALL_NUMBER 1e-13
#define MAX_ITER 300

/**
 * @brief Transposes a given matrix.
 *
 * @param mat Input matrix (flattened).
 * @param n Number of rows.
 * @param k Number of columns.
 * @return Pointer to the transposed matrix, or NULL if memory allocation fails.
 */
static double *transpose_matrix(const double *mat, int n, int k){
    int i, j;
    double *mat_t = malloc((size_t)n*k*sizeof(double));
    if(!mat_t) return NULL;
    for(i=0; i<n; i++){
        for(j=0; j<k; j++){
            mat_t[(j*n)+i] = mat[(i*k)+j];
        }
    }
    return mat_t;
}

/**
 * @brief Multiplies two matrices.
 *
 * @param matA First matrix.
 * @param matB Second matrix.
 * @param rowsA Number of rows in matA.
 * @param cols_rows Number of columns in matA and rows in matB.
 * @param colsB Number of columns in matB.
 * @return A pointer to product matrix, or NULL if memory allocation fails.
 */
static double *mat_mult(const double *matA, const double *matB, int rowsA, int cols_rows, int colsB){
    int i, j, k;
    double sum;
    double *prod = malloc(rowsA * colsB * sizeof(double));
    if(!prod) return NULL;
    for(i=0; i<rowsA; i++){
        for(j=0; j<colsB; j++){
            sum=0.0;
            for(k=0; k<cols_rows; k++){
                sum += matA[(i*cols_rows) + k]*matB[(k*colsB) + j];
            }
            prod[(i*colsB) + j] = sum;
        }
    }
    return prod;
}

/**
 * @brief Checks for convergence of the H matrix.
 *
 * @param h_new New H matrix.
 * @param h_old Previous H matrix.
 * @param n Number of rows.
 * @param k Number of columns.
 * @return 1 iff converged, 0 otherwise.
 */
static int check_convergence(const double *h_new, const double *h_old, int n, int k){
    int i, j;
    double sum = 0.0;
    for(i=0; i<n; i++){
        for(j=0; j<k; j++){
            sum += pow(h_new[(i*k) + j] - h_old[(i*k) + j], 2);
        }
    }
    return sum < EPSILON;
}

/**
 * @brief Calculates the matrix product H*Hᵀ*H.
 *
 * @param mat Input matrix.
 * @param n Number of rows.
 * @param k Number of columns.
 * @return A pointer to the  product matrix, or NULL if memory allocation fails.
 */
static double *mult_transpose(const double *mat, int n, int k){
    double *transpose, *res, *temp;
    transpose = transpose_matrix(mat, n, k);
    if(!transpose) return NULL;
    temp = mat_mult(transpose, mat, k, n, k);
    if(!temp){
        free(transpose);
        return NULL;
    }
    res = mat_mult(mat, temp, n, k, k);
    free(transpose);
    free(temp);
    return res;
}

/**
 * @brief Creates a new H matrix using the iterative update rule.
 *
 * @param old_h Previous H matrix.
 * @param w_matrix  Norm matrix.
 * @param n Number of rows (& columns in the norm matrix).
 * @param k Number of columns in H.
 * @return A pointer to the new H matrix, or NULL if memory allocation fails.
 */
static double *create_h_new(const double *old_h, const double *w_matrix, int n, int k){
    int i, j;
    double inside;
    double *h_matrix, *mone, *mehane;
    h_matrix = malloc((size_t)n*k*sizeof(double));
    if(!h_matrix) return NULL;
    mone = mat_mult(w_matrix, old_h, n, n, k);
    if(!mone){
        free(h_matrix);
        return NULL;
    }
    mehane = mult_transpose(old_h, n, k);
    if(!mehane){
        free(h_matrix);
        free(mone);
        return NULL;
    }
    for(i=0; i<n; i++){ /*calc the new H matrix based on the numerator and denominator calculated before, and BETA.*/
        for(j=0; j<k; j++){
            inside = (mone[(i*k) + j])/(mehane[(i*k) + j] + SMALL_NUMBER); /*SMALL_NUMBER is used to make sure no division by zero is done.*/
            h_matrix[(i*k) + j] = old_h[(i*k) + j]*((1.0-BETA)+(BETA*inside));
        }
    }
    free(mone);
    free(mehane);
    return h_matrix;
}

/**
 * @brief Iteratively updates the H matrix until convergence or max iterations are reached.
 *
 * @param h_matrix Initial H matrix.
 * @param w_matrix Norm matrix.
 * @param n Number of rows (& columns in the norm matrix).
 * @param k Number of columns in H.
 * @return The converged H matrix, or NULL if memory allocation fails.
 */
double *converge_h(double *h_matrix, const double *w_matrix, int n, int k){
    int i, res;
    double *old_h;
    for(i=0; i<MAX_ITER; i++){
        old_h = h_matrix;
        h_matrix = create_h_new(old_h, w_matrix, n, k);
        if(!h_matrix){
            free(old_h);
            return NULL;
        }
        res = check_convergence(h_matrix, old_h, n, k);
        free(old_h);
        if(res) break;
    }
    return h_matrix;
}

/**
 * @brief Builds the normalized similarity matrix W.
 *
 * @param d_matrix Diagonal Degree matrix.
 * @param a_matrix Similarity matrix.
 * @param n Number of data points.
 * @return A pointer to the norm matrix, or NULL if memory allocation fails.
 */
double *build_w(const double *d_matrix, const double *a_matrix, int n){
    int i,j;
    double d_i, d_j;
    double *w = malloc((size_t)n*n*sizeof(double));
    if(!w) return NULL;
    for(i=0; i<n;i++){
        d_i = d_matrix[i] != 0.0 ? 1.0/sqrt(d_matrix[i]) : 0.0;
        for(j=0;j<n;j++){
            d_j = d_matrix[j] != 0.0 ? 1.0/sqrt(d_matrix[j]) : 0.0;
            w[(i*n)+j] = d_i*a_matrix[i*n+j]*d_j; /*smart matrix multiplication of symmetric and diagonal matrices*/
        }
    }
    return w;
}

/**
 * @brief Computes the similarity value between two data points.
 *
 * @param point1 First point.
 * @param point2 Second point.
 * @param d Dimension of the points.
 * @return The value for the similarity matrix entry.
 */
static double sym_val(const double *point1, const double *point2, int d){   
    int i;
    double e_exponent=0.0, sum=0.0;
    double temp;
    
    for(i=0; i<d; i++){
       temp =  point1[i] - point2[i];
       sum += pow(temp,2);
    }
    e_exponent = (-0.5)*sum;
    return (exp(e_exponent));
}

/**
 * @brief Builds the similarity matrix A.
 *
 * @param p_matrix Flattened 1D array of size n*d of the data points.
 * @param n Number of data points.
 * @param d Dimension of the data points.
 * @return A pointer to the similarity matrix, or NULL if memory allocation fails.
 */
double *build_a(const double *p_matrix, int n, int d){
    int i, j;
    double sym_v;
    double *a = malloc((size_t)n*n*sizeof(double));
    if(!a) return NULL;
    for(i=0; i<n;i++){
        for(j=0;j<i;j++){ /*calc sym value only under the diag*/
            sym_v = sym_val(&p_matrix[i * d], &p_matrix[j * d], d); 
            a[(i*n) + j] = sym_v; 
            a[(j*n) + i] = sym_v; /*a is symmetric*/
        } 
        a[(i*n) + i] = 0.0;  /*diag is all 0's */
    }  
    return a;      
    
}

/**
 * @brief Builds the diagonal degree matrix D (diagonal, size n).
 *
 * @param a_matrix The similarity matrix.
 * @param n Number of data points.
 * @return A pointer to a 1D array of the diagonal elements of D, or NULL if memory allocation fails.
 */
double *build_d(const double *a_matrix, int n){
    int i, j;
    double d_i;
    double *d = malloc((size_t)n*sizeof(double));
    if(!d) return NULL;
    for(i=0; i<n;i++){
        d_i = 0.0;
        for(j=0;j<n;j++){ 
            d_i+=a_matrix[(i*n) + j];
        } 
        d[i] = d_i;  
    }  
    return d;
}

/**All functions responsible for running the file as a standalone.
 * when compiling, need to use -DBUILD_STANDALONE to make sure it's included
*/
#ifdef BUILD_STANDALONE

/**
 * @brief Compute a matrix on input data based on the specified goal.
 *
 * @param p_matrix Pointer to a flattened matrix of size n*d.
 * @param n Number of data points.
 * @param d Dimension of the data points.
 * @param goal A string indicating the matrix to compute: "sym", "ddg", or "norm".
 * @return Pointer to the resulting matrix (flattened), or NULL if memory allocation fails.
 */
static double *calc_mat(double *p_matrix, int n, int d, char *goal){
    double *a_mat, *d_mat, *w_mat;
    a_mat = build_a(p_matrix, n, d);
    if(!a_mat) return NULL;
    if(strcmp(goal, "sym") == 0) return a_mat;
    d_mat = build_d(a_mat, n);
    if(!d_mat){
        free(a_mat);
        return NULL;
    }
    if(strcmp(goal, "ddg") == 0){
        free(a_mat);
        return d_mat;
    }
    w_mat = build_w(d_mat, a_mat, n);
    free(a_mat);
    free(d_mat);
    if(!w_mat) return NULL;
    return w_mat;
}

/**
 * @brief Print matrix to standard output.
 *
 * @param mat Matrix to be printed (as flattened 1D array).
 * @param numRows Number of rows.
 * @param numCols Number of columns.
 */
static void print_matrix(double *mat, int numRows, int numCols){
    int i, j;
    for(i=0; i<numRows;i++){
        for(j=0;j<numCols;j++){ 
            printf("%.4f",mat[(i*numCols) + j]);
            if (j<numCols-1)
            {
                printf(",");
            }
        } 
        printf("\n");   
    }
}

/**
 * @brief Prints the Diagonal Degree matrix given its diagonal elements.
 * @param d_mat A 1D array containing the diagonal elements of D.
 * @param n Number of data points.
 */
static void print_d(double *d_mat, int n){
    int i, j;
    for(i=0; i<n;i++){
        for(j=0;j<n;j++){ 
            if(i==j) printf("%.4f",d_mat[i]);
            else printf("0.0000");
            if (j<n-1) printf(",");
        } 
        printf("\n");   
    }
}

/**
 * @brief Parse input file to determine number of points and their dimensions.
 *
 * @param fp File pointer to the input data file.
 * @param n Pointer to an integer where the number of data points will be stored.
 * @param d Pointer to an integer where the dimension of the data points will be stored.
 */
static void get_n_d(FILE *fp, int *n, int *d){
    int firstline=1;
    double p;
    char c;
    *n=0;
    *d=0;
    while (fscanf(fp, "%lf%c", &p, &c) == 2)
    {
        if(firstline) (*d)++;
        if(c == '\n'){
            (*n)++;
            firstline=0;
        }
        
    }
}

/**
 * @brief Reads data points from a file into a flat matrix.
 *
 * @param fp File pointer to the input data file.
 * @param n Number of data points to read.
 * @param d Dimension of the data points to read.
 * @return A pointer to the flat matrix of data points, or NULL if memory allocation or file reading fails.
 */
static double *read_from_file(FILE *fp, int n, int d){
    double *p_matrix;
    int i, j;
    p_matrix = malloc((size_t)n*d*sizeof(double));
    if(!p_matrix) return NULL; 
    for(i=0; i<n; i++){
        for(j=0;j<d;j++){
            if(fscanf(fp, "%lf", &p_matrix[(i*d) + j]) != 1){
                free(p_matrix);
                return NULL;
            }
            if (j< d-1) fgetc(fp);
        }
        fgetc(fp);
    }
    return p_matrix;
}

int main(int argc, char **argv){
    double *p_mat, *end_mat;
    int n, d;
    FILE *fp;
    (void)argc;
    fp = fopen(argv[2], "r");
    if(!fp) goto error_case;
    get_n_d(fp, &n, &d);
    rewind(fp); /*make sure we start reading from the beginning of the file*/
    p_mat = read_from_file(fp, n, d);
    if(!p_mat) goto error_case;
    end_mat = calc_mat(p_mat, n, d, argv[1]);
    free(p_mat);
    if(end_mat){
        if(strcmp(argv[1], "ddg") == 0) print_d(end_mat, n);
        else print_matrix(end_mat, n, n);
        free(end_mat);
        fclose(fp);
        return 0;
    }
error_case:
    printf("An Error Has Occurred\n");
    if(fp) fclose(fp);
    return 1;
}
#endif