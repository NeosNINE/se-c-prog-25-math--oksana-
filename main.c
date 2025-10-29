#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

Matrix alloc_matrix(int r, int c) {
    Matrix m;
    m.rows = r;
    m.cols = c;
    m.data = malloc(r * sizeof(float *));
    for (int i = 0; i < r; i++)
        m.data[i] = malloc(c * sizeof(float));
    return m;
}

void free_matrix(Matrix m) {
    for (int i = 0; i < m.rows; i++)
        free(m.data[i]);
    free(m.data);
}

Matrix read_matrix(FILE *f) {
    int r, c;
    if (fscanf(f, "%d %d", &r, &c) != 2)
        exit(1);
    Matrix m = alloc_matrix(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (fscanf(f, "%f", &m.data[i][j]) != 1)
                exit(1);
    return m;
}

void print_matrix(FILE *out, Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fprintf(out, "%g", m.data[i][j]);
            if (j + 1 < m.cols) fprintf(out, " ");
        }
        fprintf(out, "\n");
    }
}

Matrix add_matrix(Matrix a, Matrix b, int sign) {
    if (a.rows != b.rows || a.cols != b.cols) {
        Matrix empty = {0,0,NULL};
        return empty;
    }
    Matrix r = alloc_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            r.data[i][j] = a.data[i][j] + sign * b.data[i][j];
    return r;
}

Matrix mul_matrix(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        Matrix empty = {0,0,NULL};
        return empty;
    }
    Matrix r = alloc_matrix(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            r.data[i][j] = 0;
            for (int k = 0; k < a.cols; k++)
                r.data[i][j] += a.data[i][k] * b.data[k][j];
        }
    return r;
}

float det(Matrix m) {
    if (m.rows != m.cols)
        return NAN;
    int n = m.rows;
    float **a = malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        a[i] = malloc(n * sizeof(float));
        for (int j = 0; j < n; j++)
            a[i][j] = m.data[i][j];
    }
    float det = 1;
    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++)
            if (fabs(a[j][i]) > fabs(a[pivot][i]))
                pivot = j;
        if (fabs(a[pivot][i]) < 1e-9) {
            det = 0;
            break;
        }
        if (pivot != i) {
            float *tmp = a[i];
            a[i] = a[pivot];
            a[pivot] = tmp;
            det = -det;
        }
        det *= a[i][i];
        for (int j = i + 1; j < n; j++) {
            float f = a[j][i] / a[i][i];
            for (int k = i; k < n; k++)
                a[j][k] -= f * a[i][k];
        }
    }
    for (int i = 0; i < n; i++)
        free(a[i]);
    free(a);
    return det;
}

Matrix identity(int n) {
    Matrix I = alloc_matrix(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            I.data[i][j] = (i == j) ? 1 : 0;
    return I;
}

Matrix power_matrix(Matrix m, int p) {
    if (m.rows != m.cols || p < 0) {
        Matrix empty = {0,0,NULL};
        return empty;
    }
    if (p == 0)
        return identity(m.rows);
    Matrix result = alloc_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = m.data[i][j];
    for (int i = 1; i < p; i++) {
        Matrix temp = mul_matrix(result, m);
        free_matrix(result);
        result = temp;
    }
    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) {
        fprintf(stderr, "Cannot open input file\n");
        return 1;
    }

    FILE *out = fopen(argv[2], "w");
    if (!out) {
        fclose(in);
        fprintf(stderr, "Cannot open output file\n");
        return 1;
    }

    char op;
    if (fscanf(in, " %c", &op) != 1) {
        fprintf(stderr, "Invalid input\n");
        fclose(in); fclose(out);
        return 1;
    }

    Matrix A = read_matrix(in);
    Matrix B;
    Matrix result;
    int p;

    switch (op) {
        case '+':
        case '-':
            B = read_matrix(in);
            result = add_matrix(A, B, op == '+' ? 1 : -1);
            if (result.data == NULL) fprintf(out, "no solution\n");
            else { print_matrix(out, result); free_matrix(result); }
            free_matrix(B);
            break;
        case '*':
            B = read_matrix(in);
            result = mul_matrix(A, B);
            if (result.data == NULL) fprintf(out, "no solution\n");
            else { print_matrix(out, result); free_matrix(result); }
            free_matrix(B);
            break;
        case '^':
            if (fscanf(in, "%d", &p) != 1) {
                fprintf(stderr, "Invalid power\n");
                fclose(in); fclose(out);
                return 1;
            }
            result = power_matrix(A, p);
            if (result.data == NULL) fprintf(out, "no solution\n");
            else { print_matrix(out, result); free_matrix(result); }
            break;
        case '|': {
            float d = det(A);
            if (isnan(d)) fprintf(out, "no solution\n");
            else fprintf(out, "%g\n", d);
            break;
        }
        default:
            fprintf(out, "no solution\n");
    }

    free_matrix(A);
    fclose(in);
    fclose(out);
    return 0;
}
