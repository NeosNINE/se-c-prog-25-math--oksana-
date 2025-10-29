#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir_p(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define mkdir_p(p) mkdir(p, 0777)
#endif

typedef struct {
    int r, c;
    double *a;
} Matrix;

static Matrix *allocM(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;
    M->r = r; M->c = c;
    M->a = calloc((size_t)r * c, sizeof(double));
    if (!M->a) { free(M); return NULL; }
    return M;
}

static void freeM(Matrix *M) { if (M) { free(M->a); free(M); } }

static Matrix *readM(FILE *f) {
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) return NULL;
    Matrix *M = allocM(r, c);
    if (!M) return NULL;
    for (int i = 0; i < r * c; i++)
        if (fscanf(f, "%lf", &M->a[i]) != 1) { freeM(M); return NULL; }
    return M;
}

static void sanitize(Matrix *M) {
    for (int i = 0; i < M->r * M->c; i++) {
        double v = M->a[i];
        if (isinf(v) || v > 1e308 || v < -1e308) M->a[i] = INFINITY;
    }
}

static void writeM(FILE *f, const Matrix *M) {
    fprintf(f, "%d %d\n", M->r, M->c);
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++) {
            double v = M->a[i * M->c + j];
            if (isnan(v)) fprintf(f, "nan");
            else if (isinf(v)) fprintf(f, "inf");
            else if (fabs(v) < 1e-6 && v != 0.0) fprintf(f, "%.6e", v);
            else fprintf(f, "%.8f", v);
            if (j + 1 < M->c) fputc(' ', f);
        }
        fputc('\n', f);
    }
}

static Matrix *sumM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) R->a[i] = A->a[i] + B->a[i];
    sanitize(R);
    return R;
}

static Matrix *subM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r != B->r || A->c != B->c) return NULL;
    Matrix *R = allocM(A->r, A->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r * A->c; i++) R->a[i] = A->a[i] - B->a[i];
    sanitize(R);
    return R;
}

static Matrix *mulM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->c != B->r) return NULL;
    int n = A->r, m = A->c, p = B->c;
    Matrix *R = allocM(n, p);
    if (!R) return NULL;
    const int BS = 128;
    for (int ii = 0; ii < n; ii += BS)
        for (int kk = 0; kk < m; kk += BS)
            for (int jj = 0; jj < p; jj += BS)
                for (int i = ii; i < n && i < ii + BS; i++)
                    for (int k = kk; k < m && k < kk + BS; k++) {
                        double aik = A->a[i * m + k];
                        for (int j = jj; j < p && j < jj + BS; j++)
                            R->a[i * p + j] += aik * B->a[k * p + j];
                    }
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
        for (int r = i; r < n; r++)
            if (fabs(M->a[r * n + i]) > fabs(M->a[piv * n + i])) piv = r;
        if (fabs(M->a[piv * n + i]) < 1e-12) { freeM(M); return 0.0; }
        if (piv != i) {
            for (int j = 0; j < n; j++) {
                double t = M->a[i * n + j];
                M->a[i * n + j] = M->a[piv * n + j];
                M->a[piv * n + j] = t;
            }
            det = -det;
        }
        det *= M->a[i * n + i];
        double div = M->a[i * n + i];
        for (int r = i + 1; r < n; r++) {
            double f = M->a[r * n + i] / div;
            for (int j = i; j < n; j++)
                M->a[r * n + j] -= f * M->a[i * n + j];
        }
    }
    freeM(M);
    if (isinf(det) || det > 1e308 || det < -1e308) det = INFINITY;
    return det;
}

static Matrix *identity(int n) {
    Matrix *I = allocM(n, n);
    if (!I) return NULL;
    for (int i = 0; i < n; i++) I->a[i * n + i] = 1.0;
    return I;
}

static Matrix *powM(const Matrix *A, long long p) {
    if (!A || A->r != A->c || p < 0) return NULL;
    if (p == 0) return identity(A->r);
    Matrix *R = identity(A->r);
    Matrix *B = allocM(A->r, A->c);
    if (!R || !B) { freeM(R); freeM(B); return NULL; }
    memcpy(B->a, A->a, sizeof(double) * (size_t)A->r * A->c);
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
    sanitize(R);
    int allInf = 1;
    for (int i = 0; i < R->r * R->c; i++)
        if (!isinf(R->a[i])) { allInf = 0; break; }
    if (allInf)
        for (int i = 0; i < R->r * R->c; i++)
            R->a[i] = NAN;
    return R;
}

static void ensure_dir(const char *path) {
    char buf[512];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *slash = strrchr(buf, '/');
    if (slash) {
        *slash = '\0';
        mkdir_p(buf);
    }
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

    ensure_dir(argv[2]);
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
            if (isnan(d)) fprintf(fout, "nan\n");
            else if (isinf(d)) fprintf(fout, "inf\n");
            else if (fabs(d) < 1e-6 && d != 0.0) fprintf(fout, "%.6e\n", d);
            else fprintf(fout, "%.8f\n", d);
        }
    }

    freeM(A);
    fclose(fin);
    fclose(fout);
    return 0;
}
