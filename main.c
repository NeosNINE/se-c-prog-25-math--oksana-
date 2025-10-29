#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int r, c;
    float *a;
} Matrix;

static Matrix *allocM(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;
    M->r = r; M->c = c;
    M->a = calloc((size_t)r * c, sizeof(float));
    if (!M->a) { free(M); return NULL; }
    return M;
}

static void freeM(Matrix *M) { if (M) { free(M->a); free(M); } }

static Matrix *readM(FILE *f) {
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) return NULL;
    Matrix *M = allocM(r, c);
    if (!M) return NULL;
    for (int i = 0; i < r * c; i++) {
        double v;
        if (fscanf(f, "%lf", &v) != 1) { freeM(M); return NULL; }
        M->a[i] = (float)v;
    }
    return M;
}

static void writeM(FILE *f, const Matrix *M) {
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++) {
            double v = (double)M->a[i * M->c + j];
            if (j) fputc(' ', f);
            fprintf(f, "%g", v);
        }
        fputc('\n', f);
    }
}

static Matrix *sumM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) R->a[i] = A->a[i] + B->a[i];
    return R;
}

static Matrix *subM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) R->a[i] = A->a[i] - B->a[i];
    return R;
}

static Matrix *mulM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->c != B->r) return NULL;
    int n = A->r, m = A->c, p = B->c;
    Matrix *R = allocM(n, p);
    if (!R) return NULL;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            int pos_inf = 0;
            int neg_inf = 0;
            for (int k = 0; k < m; k++) {
                double a = (double)A->a[i * m + k];
                double b = (double)B->a[k * p + j];
                if ((a == 0.0 || b == 0.0) && (isinf(a) || isinf(b))) {
                    continue;
                }
                double term = a * b;
                if (isnan(term)) {
                    pos_inf = 1;
                    neg_inf = 1;
                    break;
                }
                if (isinf(term)) {
                    if (term > 0) pos_inf = 1;
                    else neg_inf = 1;
                    continue;
                }
                double new_sum = sum + term;
                if (isnan(new_sum)) {
                    pos_inf = 1;
                    neg_inf = 1;
                    break;
                }
                if (isinf(new_sum)) {
                    if (new_sum > 0) pos_inf = 1;
                    else neg_inf = 1;
                    continue;
                }
                sum = new_sum;
            }
            float result;
            if (pos_inf && neg_inf) result = NAN;
            else if (pos_inf) result = INFINITY;
            else if (neg_inf) result = -INFINITY;
            else result = (float)sum;
            R->a[i * p + j] = result;
        }
    }
    return R;
}

static double detM(const Matrix *A) {
    if (!A || A->r != A->c) return NAN;
    int n = A->r;
    if (n == 0) return 1.0;
    double *M = malloc(sizeof(double) * (size_t)n * n);
    if (!M) return NAN;
    for (int i = 0; i < n * n; i++)
        M[i] = (double)A->a[i];
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        int piv = i;
        for (int r = i; r < n; r++)
            if (fabs(M[r * n + i]) > fabs(M[piv * n + i])) piv = r;
        if (fabs(M[piv * n + i]) < 1e-12) { free(M); return 0.0; }
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                double t = M[i * n + j];
                M[i * n + j] = M[piv * n + j];
                M[piv * n + j] = t;
            }
            det = -det;
        }
        det *= M[i * n + i];
        double div = M[i * n + i];
        for (int r = i + 1; r < n; r++) {
            double f = M[r * n + i] / div;
            for (int j = i; j < n; j++)
                M[r * n + j] -= f * M[i * n + j];
        }
    }
    free(M);
    if (isinf(det) || det > 1e308 || det < -1e308) det = INFINITY;
    return det;
}

static Matrix *identity(int n) {
    Matrix *I = allocM(n, n);
    if (!I) return NULL;
    for (int i = 0; i < n; i++) I->a[i * n + i] = 1.0f;
    return I;
}

static Matrix *powM(const Matrix *A, long long p) {
    if (!A || A->r != A->c || p < 0) return NULL;
    if (p == 0) return identity(A->r);
    Matrix *R = identity(A->r);
    Matrix *B = allocM(A->r, A->c);
    if (!R || !B) { freeM(R); freeM(B); return NULL; }
    memcpy(B->a, A->a, sizeof(float) * (size_t)A->r * A->c);
    while (p > 0) {
        if (p & 1) {
            Matrix *tmp = mulM(R, B);
            freeM(R);
            R = tmp;
            if (!R) { freeM(B); return NULL; }
        }
        p >>= 1;
        if (p) {
            Matrix *tmp = mulM(B, B);
            freeM(B);
            B = tmp;
            if (!B) { freeM(R); return NULL; }
        }
    }
    freeM(B);
    return R;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Error: wrong argument count\n");
        fflush(stderr);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file\n");
        fflush(stderr);
        return 1;
    }

    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        fprintf(stderr, "Error: cannot create output file\n");
        fflush(stderr);
        fclose(fin);
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fprintf(stderr, "Error: cannot read operator\n");
        fflush(stderr);
        fclose(fin);
        fclose(fout);
        return 1;
    }

    if (op != '+' && op != '-' && op != '*' && op != '^' && op != '|') {
        fprintf(stderr, "Error: unknown operator\n");
        fflush(stderr);
        fclose(fin);
        fclose(fout);
        return 1;
    }

    Matrix *A = readM(fin);
    if (!A) {
        fprintf(fout, "no solution\n");
        fclose(fout);
        fclose(fin);
        return 0;
    }

    if (op == '+') {
        Matrix *B = readM(fin);
        Matrix *R = (B ? sumM(A, B) : NULL);
        if (!R) fprintf(fout, "no solution\n");
        else { writeM(fout, R); freeM(R); }
        freeM(B);
    } else if (op == '-') {
        Matrix *B = readM(fin);
        Matrix *R = (B ? subM(A, B) : NULL);
        if (!R) fprintf(fout, "no solution\n");
        else { writeM(fout, R); freeM(R); }
        freeM(B);
    } else if (op == '*') {
        Matrix *B = readM(fin);
        Matrix *R = (B ? mulM(A, B) : NULL);
        if (!R) fprintf(fout, "no solution\n");
        else { writeM(fout, R); freeM(R); }
        freeM(B);
    } else if (op == '^') {
        long long p;
        if (fscanf(fin, "%lld", &p) != 1)
            fprintf(fout, "no solution\n");
        else {
            Matrix *R = powM(A, p);
            if (!R) fprintf(fout, "no solution\n");
            else { writeM(fout, R); freeM(R); }
        }
    } else if (op == '|') {
        if (A->r != A->c)
            fprintf(fout, "no solution\n");
        else {
            double d = detM(A);
            fprintf(fout, "%g\n", d);
        }
    }

    freeM(A);
    fclose(fin);
    fclose(fout);
    return 0;
}
