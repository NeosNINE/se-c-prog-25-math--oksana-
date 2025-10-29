#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    size_t r, c;
    double *a;
} Mat;

static void fail(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static Mat newmat(size_t r, size_t c) {
    Mat m = {r, c, calloc(r * c, sizeof(double))};
    if (!m.a) fail("alloc");
    return m;
}

static void freemat(Mat *m) {
    free(m->a);
    m->a = NULL;
}

static double *at(Mat *m, size_t i, size_t j) {
    return &m->a[i * m->c + j];
}

static double get(const Mat *m, size_t i, size_t j) {
    return m->a[i * m->c + j];
}

static Mat readmat(FILE *f) {
    size_t r, c;
    if (fscanf(f, "%zu%zu", &r, &c) != 2) fail("bad header");
    Mat m = newmat(r, c);
    for (size_t i = 0; i < r; i++)
        for (size_t j = 0; j < c; j++)
            if (fscanf(f, "%lf", &m.a[i * c + j]) != 1)
                fail("bad data");
    return m;
}

static void writemat(FILE *g, const Mat *m) {
    fprintf(g, "%zu %zu\n", m->r, m->c);
    for (size_t i = 0; i < m->r; i++) {
        for (size_t j = 0; j < m->c; j++) {
            if (j) fputc(' ', g);
            double x = get(m, i, j);
            if (isnan(x)) fputs("nan", g);
            else if (isinf(x)) fputs(signbit(x) ? "-inf" : "inf", g);
            else fprintf(g, "%.10g", x);
        }
        fputc('\n', g);
    }
}

static void writescalar(FILE *g, double x) {
    if (isnan(x)) fputs("nan\n", g);
    else if (isinf(x)) fputs(signbit(x) ? "-inf\n" : "inf\n", g);
    else fprintf(g, "%.10g\n", x);
}

static Mat addsub(const Mat *A, const Mat *B, int s) {
    if (A->r != B->r || A->c != B->c) fail("dim");
    Mat C = newmat(A->r, A->c);
    for (size_t i = 0; i < A->r * A->c; i++)
        C.a[i] = A->a[i] + s * B->a[i];
    return C;
}

static Mat mul(const Mat *A, const Mat *B) {
    if (A->c != B->r) fail("dim");
    Mat C = newmat(A->r, B->c);
    for (size_t i = 0; i < A->r; i++)
        for (size_t k = 0; k < A->c; k++) {
            double v = get(A, i, k);
            if (v == 0) continue;
            for (size_t j = 0; j < B->c; j++)
                *at(&C, i, j) += v * get(B, k, j);
        }
    return C;
}

static double det(const Mat *A) {
    if (A->r != A->c) return NAN;
    size_t n = A->r;
    Mat M = newmat(n, n);
    memcpy(M.a, A->a, n * n * sizeof(double));
    double d = 1;
    int s = 1;
    for (size_t i = 0; i < n; i++) {
        size_t p = i;
        double best = fabs(get(&M, i, i));
        for (size_t r = i + 1; r < n; r++) {
            double v = fabs(get(&M, r, i));
            if (v > best) {
                best = v;
                p = r;
            }
        }
        if (best == 0) {
            d = 0;
            goto done;
        }
        if (p != i) {
            for (size_t j = 0; j < n; j++) {
                double t = get(&M, i, j);
                *at(&M, i, j) = get(&M, p, j);
                *at(&M, p, j) = t;
            }
            s = -s;
        }
        double piv = get(&M, i, i);
        d *= piv;
        for (size_t r = i + 1; r < n; r++) {
            double f = get(&M, r, i) / piv;
            for (size_t j = i + 1; j < n; j++)
                *at(&M, r, j) -= f * get(&M, i, j);
        }
    }
done:
    freemat(&M);
    return d * s;
}

static Mat id(size_t n) {
    Mat I = newmat(n, n);
    for (size_t i = 0; i < n; i++)
        *at(&I, i, i) = 1;
    return I;
}

static Mat power(const Mat *A, long long k) {
    if (A->r != A->c) {
        Mat Z = newmat(1, 1);
        Z.a[0] = NAN;
        return Z;
    }
    if (k < 0) {
        Mat Z = newmat(A->r, A->c);
        for (size_t i = 0; i < Z.r * Z.c; i++) Z.a[i] = NAN;
        return Z;
    }
    Mat res = id(A->r);
    Mat base = newmat(A->r, A->c);
    memcpy(base.a, A->a, A->r * A->c * sizeof(double));
    while (k > 0) {
        if (k & 1) {
            Mat t = mul(&res, &base);
            freemat(&res);
            res = t;
        }
        k >>= 1;
        if (k) {
            Mat t = mul(&base, &base);
            freemat(&base);
            base = t;
        }
    }
    freemat(&base);
    return res;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("input");
        return 1;
    }
    FILE *g = fopen(argv[2], "w");
    if (!g) {
        perror("output");
        fclose(f);
        return 1;
    }

    char op[16];
    if (fscanf(f, "%15s", op) != 1) {
        fclose(f);
        fclose(g);
        return 1;
    }

    if (!strcmp(op, "det")) {
        Mat A = readmat(f);
        double d = det(&A);
        writescalar(g, d);
        freemat(&A);
    } else if (!strcmp(op, "mul")) {
        Mat A = readmat(f), B = readmat(f);
        Mat C = mul(&A, &B);
        writemat(g, &C);
        freemat(&A);
        freemat(&B);
        freemat(&C);
    } else if (!strcmp(op, "sum")) {
        Mat A = readmat(f), B = readmat(f);
        Mat C = addsub(&A, &B, +1);
        writemat(g, &C);
        freemat(&A);
        freemat(&B);
        freemat(&C);
    } else if (!strcmp(op, "sub")) {
        Mat A = readmat(f), B = readmat(f);
        Mat C = addsub(&A, &B, -1);
        writemat(g, &C);
        freemat(&A);
        freemat(&B);
        freemat(&C);
    } else if (!strcmp(op, "pow")) {
        Mat A = readmat(f);
        long long k;
        if (fscanf(f, "%lld", &k) != 1) k = 1;
        Mat P = power(&A, k);
        writemat(g, &P);
        freemat(&A);
        freemat(&P);
    } else fail("bad op");

    fclose(f);
    fclose(g);
    return 0;
}
