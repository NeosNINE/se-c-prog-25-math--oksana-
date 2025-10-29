#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int r, c;
    double *a;
} Matrix;

static Matrix *allocM(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;
    M->r = r;
    M->c = c;
    M->a = calloc((size_t)r * c, sizeof(double));
    if (!M->a) {
        free(M);
        return NULL;
    }
    return M;
}

static void freeM(Matrix *M) {
    if (M) {
        free(M->a);
        free(M);
    }
}

static Matrix *readM(FILE *f, bool *parse_error) {
    if (parse_error) *parse_error = false;
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) {
        if (parse_error) *parse_error = true;
        return NULL;
    }
    Matrix *M = allocM(r, c);
    if (!M) {
        if (parse_error) *parse_error = true;
        return NULL;
    }
    for (int i = 0; i < r * c; i++) {
        if (fscanf(f, "%lf", &M->a[i]) != 1) {
            if (parse_error) *parse_error = true;
            freeM(M);
            return NULL;
        }
    }
    return M;
}

static void sanitize(Matrix *M) {
    if (!M) return;
    for (int i = 0; i < M->r * M->c; i++) {
        double v = M->a[i];
        if (isnan(v)) {
            M->a[i] = INFINITY;
        } else if (isinf(v)) {
            M->a[i] = copysign(INFINITY, v);
        } else if (v > 1e308) {
            M->a[i] = INFINITY;
        } else if (v < -1e308) {
            M->a[i] = -INFINITY;
        }
    }
}

static void writeM(FILE *f, const Matrix *M) {
    fprintf(f, "%d %d\n", M->r, M->c);
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++) {
            double v = M->a[i * M->c + j];
            if (isnan(v)) {
                fputs("nan", f);
            } else if (isinf(v)) {
                if (signbit(v)) fputs("-inf", f);
                else fputs("inf", f);
            } else if (fabs(v) < 1e-6 && v != 0.0) {
                fprintf(f, "%.6e", v);
            } else {
                fprintf(f, "%.8f", v);
            }
            if (j + 1 < M->c) fputc(' ', f);
        }
        fputc('\n', f);
    }
}

static Matrix *sumM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) {
        R->a[i] = A->a[i] + B->a[i];
    }
    sanitize(R);
    return R;
}

static Matrix *subM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) {
        R->a[i] = A->a[i] - B->a[i];
    }
    sanitize(R);
    return R;
}

static Matrix *mulM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->c != B->r) return NULL;
    int n = A->r;
    int m = A->c;
    int p = B->c;
    Matrix *R = allocM(n, p);
    if (!R) return NULL;
    double *BT = malloc(sizeof(double) * (size_t)m * p);
    if (!BT) {
        freeM(R);
        return NULL;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            BT[j * m + i] = B->a[i * p + j];
        }
    }
    for (int i = 0; i < n; i++) {
        const double *rowA = &A->a[i * m];
        for (int j = 0; j < p; j++) {
            const double *rowB = &BT[j * m];
            double acc = 0.0;
            for (int k = 0; k < m; k++) {
                acc += rowA[k] * rowB[k];
            }
            R->a[i * p + j] = acc;
        }
    }
    free(BT);
    sanitize(R);
    return R;
}

static double detM(const Matrix *A) {
    if (!A || A->r != A->c) return NAN;
    int n = A->r;
    if (n == 0) return 1.0;
    Matrix *M = allocM(n, n);
    if (!M) return NAN;
    memcpy(M->a, A->a, sizeof(double) * (size_t)n * n);
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        int piv = i;
        for (int r = i; r < n; r++) {
            if (fabs(M->a[r * n + i]) > fabs(M->a[piv * n + i])) piv = r;
        }
        double pivot = M->a[piv * n + i];
        if (fabs(pivot) < 1e-12) {
            freeM(M);
            return 0.0;
        }
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                double t = M->a[i * n + j];
                M->a[i * n + j] = M->a[piv * n + j];
                M->a[piv * n + j] = t;
            }
            det = -det;
        }
        det *= M->a[i * n + i];
        double inv = 1.0 / M->a[i * n + i];
        for (int r = i + 1; r < n; r++) {
            double factor = M->a[r * n + i] * inv;
            if (factor == 0.0) continue;
            for (int j = i; j < n; j++) {
                M->a[r * n + j] -= factor * M->a[i * n + j];
            }
        }
    }
    freeM(M);
    if (isnan(det)) return NAN;
    if (isinf(det)) return copysign(INFINITY, det);
    if (det > 1e308) return INFINITY;
    if (det < -1e308) return -INFINITY;
    return det;
}

static Matrix *identity(int n) {
    Matrix *I = allocM(n, n);
    if (!I) return NULL;
    for (int i = 0; i < n; i++) {
        I->a[i * n + i] = 1.0;
    }
    return I;
}

static Matrix *powM(const Matrix *A, long long p) {
    if (!A || A->r != A->c || p < 0) return NULL;
    if (p == 0) return identity(A->r);
    Matrix *R = identity(A->r);
    Matrix *B = allocM(A->r, A->c);
    if (!R || !B) {
        freeM(R);
        freeM(B);
        return NULL;
    }
    memcpy(B->a, A->a, sizeof(double) * (size_t)A->r * A->c);
    long long exp = p;
    while (exp > 0) {
        if (exp & 1LL) {
            Matrix *tmp = mulM(R, B);
            freeM(R);
            R = tmp;
            if (!R) {
                freeM(B);
                return NULL;
            }
        }
        exp >>= 1;
        if (exp) {
            Matrix *tmp = mulM(B, B);
            freeM(B);
            B = tmp;
            if (!B) {
                freeM(R);
                return NULL;
            }
        }
    }
    freeM(B);
    sanitize(R);
    return R;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Error: wrong argument count\n");
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file\n");
        return 1;
    }

    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        fprintf(stderr, "Error: cannot create output file\n");
        fclose(fin);
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fprintf(stderr, "Error: cannot read operator\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    if (op != '+' && op != '-' && op != '*' && op != '^' && op != '|') {
        fprintf(stderr, "Error: unknown operator\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    bool parse_error = false;
    Matrix *A = readM(fin, &parse_error);
    if (!A) {
        if (!parse_error) fprintf(stderr, "Error: cannot read matrix\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    int exit_code = 0;
    if (op == '+') {
        Matrix *B = readM(fin, &parse_error);
        if (!B) {
            exit_code = 1;
        } else {
            Matrix *R = sumM(A, B);
            if (!R) {
                fprintf(fout, "no solution\n");
            } else {
                writeM(fout, R);
                freeM(R);
            }
        }
        freeM(B);
    } else if (op == '-') {
        Matrix *B = readM(fin, &parse_error);
        if (!B) {
            exit_code = 1;
        } else {
            Matrix *R = subM(A, B);
            if (!R) {
                fprintf(fout, "no solution\n");
            } else {
                writeM(fout, R);
                freeM(R);
            }
        }
        freeM(B);
    } else if (op == '*') {
        Matrix *B = readM(fin, &parse_error);
        if (!B) {
            exit_code = 1;
        } else {
            Matrix *R = mulM(A, B);
            if (!R) {
                fprintf(fout, "no solution\n");
            } else {
                writeM(fout, R);
                freeM(R);
            }
        }
        freeM(B);
    } else if (op == '^') {
        long long p;
        if (fscanf(fin, "%lld", &p) != 1) {
            exit_code = 1;
        } else {
            Matrix *R = powM(A, p);
            if (!R) {
                fprintf(fout, "no solution\n");
            } else {
                writeM(fout, R);
                freeM(R);
            }
        }
    } else if (op == '|') {
        if (A->r != A->c) {
            fprintf(fout, "no solution\n");
        } else {
            double d = detM(A);
            if (isnan(d)) {
                fputs("nan\n", fout);
            } else if (isinf(d)) {
                if (signbit(d)) fputs("-inf\n", fout);
                else fputs("inf\n", fout);
            } else if (fabs(d) < 1e-6 && d != 0.0) {
                fprintf(fout, "%.6e\n", d);
            } else {
                fprintf(fout, "%.8f\n", d);
            }
        }
    }

    freeM(A);
    fclose(fin);
    fclose(fout);

    if (exit_code != 0 || parse_error) {
        return 1;
    }
    return 0;
}