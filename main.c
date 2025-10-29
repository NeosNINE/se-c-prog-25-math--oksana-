#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

Matrix *create_matrix(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = r;
    m->cols = c;
    m->data = calloc(r, sizeof(double *));
    if (!m->data) { free(m); return NULL; }
    for (int i = 0; i < r; i++) {
        m->data[i] = calloc(c, sizeof(double));
        if (!m->data[i]) { for (int k=0;k<i;k++) free(m->data[k]); free(m->data); free(m); return NULL; }
    }
    return m;
}

void free_matrix(Matrix *m) {
    if (!m) return;
    for (int i = 0; i < m->rows; i++) free(m->data[i]);
    free(m->data);
    free(m);
}

int read_matrix(FILE *f, Matrix **m) {
    int r, c;
    if (fscanf(f, "%d %d", &r, &c) != 2 || r <= 0 || c <= 0) return 1;
    Matrix *tmp = create_matrix(r, c);
    if (!tmp) return 1;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (fscanf(f, "%lf", &tmp->data[i][j]) != 1) { free_matrix(tmp); return 1; }
    *m = tmp;
    return 0;
}

int write_matrix(FILE *f, Matrix *m) {
    fprintf(f, "%d %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++)
            fprintf(f, "%.6lf%c", m->data[i][j], j == m->cols-1 ? '\n' : ' ');
    }
    return 0;
}

Matrix *sum_matrix(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix *r = create_matrix(a->rows, a->cols);
    if (!r) return NULL;
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            r->data[i][j] = a->data[i][j] + b->data[i][j];
    return r;
}

Matrix *sub_matrix(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix *r = create_matrix(a->rows, a->cols);
    if (!r) return NULL;
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            r->data[i][j] = a->data[i][j] - b->data[i][j];
    return r;
}

Matrix *mul_matrix(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) return NULL;
    Matrix *r = create_matrix(a->rows, b->cols);
    if (!r) return NULL;
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < b->cols; j++) {
            long double s = 0.0;
            for (int k = 0; k < a->cols; k++)
                s += (long double)a->data[i][k] * b->data[k][j];
            r->data[i][j] = (double)s;
        }
    return r;
}

Matrix *copy_matrix(Matrix *m) {
    Matrix *r = create_matrix(m->rows, m->cols);
    if (!r) return NULL;
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            r->data[i][j] = m->data[i][j];
    return r;
}

Matrix *pow_matrix(Matrix *m, int n) {
    if (m->rows != m->cols || n < 0) return NULL;
    Matrix *res = create_matrix(m->rows, m->cols);
    Matrix *base = copy_matrix(m);
    if (!res || !base) { free_matrix(res); free_matrix(base); return NULL; }

    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            res->data[i][j] = (i == j);

    while (n > 0) {
        if (n % 2 == 1) {
            Matrix *tmp = mul_matrix(res, base);
            if (!tmp) { free_matrix(res); free_matrix(base); return NULL; }
            free_matrix(res);
            res = tmp;
        }
        n /= 2;
        if (n) {
            Matrix *tmp = mul_matrix(base, base);
            if (!tmp) { free_matrix(res); free_matrix(base); return NULL; }
            free_matrix(base);
            base = tmp;
        }
    }
    free_matrix(base);
    return res;
}

double det_matrix(Matrix *m) {
    if (m->rows != m->cols) return NAN;
    int n = m->rows;
    double det = 1.0;
    double **a = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        a[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
            a[i][j] = m->data[i][j];
    }
    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i+1; j < n; j++)
            if (fabs(a[j][i]) > fabs(a[pivot][i])) pivot = j;
        if (fabs(a[pivot][i]) < 1e-12) { det = 0; goto end; }
        if (pivot != i) { double *t = a[i]; a[i]=a[pivot]; a[pivot]=t; det = -det; }
        det *= a[i][i];
        for (int j = i+1; j < n; j++)
            a[i][j] /= a[i][i];
        for (int k = i+1; k < n; k++)
            for (int j = i+1; j < n; j++)
                a[k][j] -= a[k][i]*a[i][j];
    }
end:
    for (int i = 0; i < n; i++) free(a[i]);
    free(a);
    return det;
}

int main(int argc, char **argv) {
    if (argc != 5) return 1;
    char *op = argv[1];
    char *in1 = argv[2];
    char *in2 = argv[3];
    char *out = argv[4];

    FILE *f1 = fopen(in1, "r");
    if (!f1) return 1;
    Matrix *A = NULL, *B = NULL, *R = NULL;
    if (read_matrix(f1, &A)) { fclose(f1); return 1; }
    fclose(f1);

    FILE *f2 = NULL;
    if (strcmp(op, "det") != 0) {
        f2 = fopen(in2, "r");
        if (!f2) { free_matrix(A); return 1; }
        if (read_matrix(f2, &B)) { fclose(f2); free_matrix(A); return 1; }
        fclose(f2);
    }

    FILE *fo = fopen(out, "w");
    if (!fo) { free_matrix(A); free_matrix(B); return 1; }

    int ret = 0;
    if (strcmp(op, "sum") == 0) {
        R = sum_matrix(A, B);
        if (!R) ret = 1;
        else write_matrix(fo, R);
    } else if (strcmp(op, "sub") == 0) {
        R = sub_matrix(A, B);
        if (!R) ret = 1;
        else write_matrix(fo, R);
    } else if (strcmp(op, "mul") == 0) {
        R = mul_matrix(A, B);
        if (!R) ret = 1;
        else write_matrix(fo, R);
    } else if (strcmp(op, "pow") == 0) {
        int power = (int)strtol(in2, NULL, 10);
        R = pow_matrix(A, power);
        if (!R) ret = 1;
        else write_matrix(fo, R);
    } else if (strcmp(op, "det") == 0) {
        double d = det_matrix(A);
        if (isnan(d)) ret = 1;
        else fprintf(fo, "%.6lf\n", d);
    } else ret = 1;

    fclose(fo);
    free_matrix(A);
    free_matrix(B);
    free_matrix(R);
    return ret;
}
