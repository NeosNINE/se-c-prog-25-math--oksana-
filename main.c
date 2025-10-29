
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int r, c;
    double *a; // row-major, size r*c
} Matrix;

/* ---------- Utilities ---------- */

static void free_matrix(Matrix *M) {
    if (!M) return;
    free(M->a);
    free(M);
}

static Matrix* alloc_matrix(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = (Matrix*)calloc(1, sizeof(Matrix));
    if (!M) return NULL;
    M->r = r; M->c = c;
    M->a = (double*)calloc((size_t)r * (size_t)c, sizeof(double));
    if (!M->a) { free(M); return NULL; }
    return M;
}

static inline double get(const Matrix *M, int i, int j) {
    return M->a[(size_t)i * (size_t)M->c + (size_t)j];
}
static inline void set(Matrix *M, int i, int j, double v) {
    M->a[(size_t)i * (size_t)M->c + (size_t)j] = v;
}

/* ---------- I/O ---------- */

static int read_matrix(FILE *f, Matrix **out) {
    if (!f || !out) return -1;
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) return -2;
    if (r <= 0 || c <= 0) return -3;
    Matrix *M = alloc_matrix(r, c);
    if (!M) return -4;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double x;
            if (fscanf(f, "%lf", &x) != 1) { free_matrix(M); return -5; }
            set(M, i, j, x);
        }
    }
    *out = M;
    return 0;
}

static int write_matrix(FILE *f, const Matrix *M) {
    if (!f || !M) return -1;
    // размеры первой строкой
    if (fprintf(f, "%d %d\n", M->r, M->c) < 0) return -2;
    for (int i = 0; i < M->r; ++i) {
        for (int j = 0; j < M->c; ++j) {
            // шесть знаков после запятой — стабильно для ассёртов
            if (fprintf(f, "%.6f%s", get(M,i,j), (j+1==M->c) ? "" : " ") < 0) return -3;
        }
        if (fputc('\n', f) == EOF) return -4;
    }
    return 0;
}

/* ---------- Ops ---------- */

static Matrix* sum_matrix(const Matrix *A, const Matrix *B) {
    if (!A || !B) return NULL;
    if (A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = alloc_matrix(A->r, A->c);
    if (!R) return NULL;
    int n = A->r * A->c;
    for (int k = 0; k < n; ++k) R->a[k] = A->a[k] + B->a[k];
    return R;
}

static Matrix* sub_matrix(const Matrix *A, const Matrix *B) {
    if (!A || !B) return NULL;
    if (A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = alloc_matrix(A->r, A->c);
    if (!R) return NULL;
    int n = A->r * A->c;
    for (int k = 0; k < n; ++k) R->a[k] = A->a[k] - B->a[k];
    return R;
}

static Matrix* mul_matrix(const Matrix *A, const Matrix *B) {
    if (!A || !B) return NULL;
    if (A->c != B->r) return NULL;
    Matrix *R = alloc_matrix(A->r, B->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r; ++i) {
        for (int k = 0; k < A->c; ++k) {
            double aik = get(A, i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < B->c; ++j) {
                R->a[(size_t)i*(size_t)R->c + (size_t)j] += aik * get(B, k, j);
            }
        }
    }
    return R;
}

static Matrix* identity(int n) {
    Matrix *I = alloc_matrix(n, n);
    if (!I) return NULL;
    for (int i = 0; i < n; ++i) set(I, i, i, 1.0);
    return I;
}

static int is_square(const Matrix *A) {
    return A && A->r == A->c;
}

static double det_matrix(const Matrix *A) {
    if (!is_square(A)) return NAN;
    int n = A->r;
    // Скопировать в рабочий массив
    Matrix *M = alloc_matrix(n, n);
    if (!M) return NAN;
    memcpy(M->a, A->a, (size_t)n*(size_t)n*sizeof(double));

    double det = 1.0;
    int sign = 1;

    for (int i = 0; i < n; ++i) {
        // частичный выбор главного элемента
        int piv = i;
        double mx = fabs(get(M,i,i));
        for (int r = i+1; r < n; ++r) {
            double v = fabs(get(M,r,i));
            if (v > mx) { mx = v; piv = r; }
        }
        if (mx == 0.0) { det = 0.0; goto done; }
        if (piv != i) {
            // swap rows i <-> piv
            for (int j = 0; j < n; ++j) {
                double t = get(M,i,j);
                set(M,i,j, get(M,piv,j));
                set(M,piv,j, t);
            }
            sign = -sign;
        }
        double diag = get(M,i,i);
        det *= diag;

        // исключение ниже диагонали
        for (int r = i+1; r < n; ++r) {
            double factor = get(M,r,i) / diag;
            if (factor == 0.0) continue;
            set(M, r, i, 0.0);
            for (int j = i+1; j < n; ++j) {
                double val = get(M,r,j) - factor * get(M,i,j);
                set(M, r, j, val);
            }
        }
    }

done:
    det *= sign;
    free_matrix(M);
    return det;
}

static Matrix* pow_matrix(const Matrix *A, long long p) {
    if (!is_square(A) || p < 0) return NULL;
    int n = A->r;
    Matrix *base = alloc_matrix(n, n);
    if (!base) return NULL;
    memcpy(base->a, A->a, (size_t)n*(size_t)n*sizeof(double));
    Matrix *res = identity(n);
    if (!res) { free_matrix(base); return NULL; }
    if (p == 0) { free_matrix(base); return res; }

    while (p > 0) {
        if (p & 1LL) {
            Matrix *tmp = mul_matrix(res, base);
            if (!tmp) { free_matrix(res); free_matrix(base); return NULL; }
            free_matrix(res);
            res = tmp;
        }
        p >>= 1LL;
        if (p) {
            Matrix *sq = mul_matrix(base, base);
            if (!sq) { free_matrix(res); free_matrix(base); return NULL; }
            free_matrix(base);
            base = sq;
        }
    }
    free_matrix(base);
    return res;
}

/* ---------- CLI ---------- */

static void usage(void) {
    fprintf(stderr, "Usage: matrix <op> <input1> [input2] <output>\n");
    fprintf(stderr, "ops: sum|sub|mul|pow|det\n");
}

int main(int argc, char **argv) {
    // Ожидается 4 или 5 аргументов: prog op in1 [in2] out
    if (argc < 4 || argc > 5) { usage(); return 1; }

    const char *op  = argv[1];
    const char *in1 = argv[2];
    const char *in2 = (argc == 5) ? argv[3] : NULL;
    const char *out = (argc == 5) ? argv[4] : argv[3];

    // Валидация набора входов под операцию
    int need_two = (!strcmp(op,"sum") || !strcmp(op,"sub") || !strcmp(op,"mul"));
    int need_pow = (!strcmp(op,"pow"));
    int need_det = (!strcmp(op,"det"));

    if (need_two && !in2) { usage(); return 1; }
    if (need_pow && !in2) { usage(); return 1; }
    if (!(need_two || need_pow || need_det)) { fprintf(stderr, "Unknown op\n"); return 1; }

    // открыть вход(а)
    FILE *f1 = fopen(in1, "r");
    if (!f1) { fprintf(stderr, "Cannot open %s\n", in1); return 1; }
    Matrix *A = NULL, *B = NULL, *R = NULL;
    int rc = read_matrix(f1, &A);
    fclose(f1);
    if (rc) { fprintf(stderr, "Read error in %s\n", in1); return 1; }

    if (need_two) {
        FILE *f2 = fopen(in2, "r");
        if (!f2) { fprintf(stderr, "Cannot open %s\n", in2); free_matrix(A); return 1; }
        rc = read_matrix(f2, &B);
        fclose(f2);
        if (rc) { fprintf(stderr, "Read error in %s\n", in2); free_matrix(A); return 1; }
    }

    FILE *fo = fopen(out, "w");
    if (!fo) { fprintf(stderr, "Cannot open %s\n", out); free_matrix(A); free_matrix(B); return 1; }

    int exitcode = 0;

    if (!strcmp(op,"sum")) {
        R = sum_matrix(A, B);
        if (!R) { fprintf(stderr, "Sum failed (size mismatch?)\n"); exitcode = 1; }
        else if (write_matrix(fo, R)) { fprintf(stderr, "Write error\n"); exitcode = 1; }

    } else if (!strcmp(op,"sub")) {
        R = sub_matrix(A, B);
        if (!R) { fprintf(stderr, "Sub failed (size mismatch?)\n"); exitcode = 1; }
        else if (write_matrix(fo, R)) { fprintf(stderr, "Write error\n"); exitcode = 1; }

    } else if (!strcmp(op,"mul")) {
        R = mul_matrix(A, B);
        if (!R) { fprintf(stderr, "Mul failed (size mismatch?)\n"); exitcode = 1; }
        else if (write_matrix(fo, R)) { fprintf(stderr, "Write error\n"); exitcode = 1; }

    } else if (!strcmp(op,"pow")) {
        // in2 — файл с одной целой степенью
        FILE *pf = fopen(in2, "r");
        if (!pf) { fprintf(stderr, "Cannot open %s\n", in2); exitcode = 1; }
        else {
            long long p; int got = fscanf(pf, "%lld", &p);
            fclose(pf);
            if (got != 1) { fprintf(stderr, "Pow: invalid exponent file\n"); exitcode = 1; }
            else {
                R = pow_matrix(A, p);
                if (!R) { fprintf(stderr, "Pow failed (square matrix required, p>=0)\n"); exitcode = 1; }
                else if (write_matrix(fo, R)) { fprintf(stderr, "Write error\n"); exitcode = 1; }
            }
        }

    } else if (!strcmp(op,"det")) {
        double d = det_matrix(A);
        if (isnan(d)) { fprintf(stderr, "Det failed (square matrix required)\n"); exitcode = 1; }
        else {
            // только число детерминанта
            if (fprintf(fo, "%.6f\n", d) < 0) { fprintf(stderr, "Write error\n"); exitcode = 1; }
        }
    }

    fclose(fo);
    free_matrix(A);
    free_matrix(B);
    free_matrix(R);
    return exitcode;
}
