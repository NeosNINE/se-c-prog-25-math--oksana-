#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static void writeM(FILE *f, const Matrix *M) {
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++) {
            fprintf(f, "%g", M->a[i * M->c + j]);
            if (j < M->c - 1) fprintf(f, " ");
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
    Matrix *R = allocM(A->r, B->c);
    if (!R) return NULL;
    for (int i = 0; i < A->r; i++)
        for (int k = 0; k < A->c; k++)
            for (int j = 0; j < B->c; j++)
                R->a[i * R->c + j] += A->a[i * A->c + k] * B->a[k * B->c + j];
    return R;
}

static double detM(const Matrix *A) {
    if (!A || A->r != A->c) return NAN;
    int n = A->r;
    Matrix *M = allocM(n, n);
    if (!M) return NAN;
    memcpy(M->a, A->a, sizeof(double) * n * n);
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        int piv = i;
        for (int r = i; r < n; r++)
            if (fabs(M->a[r * n + i]) > fabs(M->a[piv * n + i])) piv = r;
        if (fabs(M->a[piv * n + i]) < 1e-12) { det = 0.0; freeM(M); return det; }
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
            for (int j = i; j < n; j++) M->a[r * n + j] -= f * M->a[i * n + j];
        }
    }
    freeM(M);
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
    Matrix *R = identity(A->r);
    if (!R) return NULL;
    Matrix *B = allocM(A->r, A->c);
    if (!B) { freeM(R); return NULL; }
    memcpy(B->a, A->a, sizeof(double) * A->r * A->c);
    while (p > 0) {
        if (p & 1) {
            Matrix *t = mulM(R, B);
            freeM(R);
            R = t;
            if (!R) { freeM(B); return NULL; }
        }
        p >>= 1;
        if (p > 0) {
            Matrix *t = mulM(B, B);
            freeM(B);
            B = t;
            if (!B) { freeM(R); return NULL; }
        }
    }
    freeM(B);
    return R;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Error: cannot open %s\n", argv[1]);
        return 1;
    }

    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        fclose(fin);
        fprintf(stderr, "Error: cannot open %s\n", argv[2]);
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fclose(fin);
        fclose(fout);
        fprintf(stderr, "Error: cannot read operator\n");
        return 1;
    }

    Matrix *A = readM(fin);
    if (!A) {
        fclose(fin);
        fclose(fout);
        fprintf(stderr, "Error: invalid first matrix\n");
        return 1;
    }

    if (op == '+' || op == '-' || op == '*') {
        Matrix *B = readM(fin);
        if (!B) {
            freeM(A);
            fclose(fin);
            fclose(fout);
            fprintf(stderr, "Error: invalid second matrix\n");
            return 1;
        }
        Matrix *R = NULL;
        if (op == '+') R = sumM(A, B);
        else if (op == '-') R = subM(A, B);
        else /* op == '*' */ R = mulM(A, B);
        freeM(A);
        freeM(B);
        if (!R) {
            fclose(fin);
            fclose(fout);
            fprintf(stderr, "Error: operation failed\n");
            return 1;
        }
        writeM(fout, R);
        freeM(R);
        fclose(fin);
        fclose(fout);
        return 0;
    } else if (op == '^') {
        long long p;
        if (fscanf(fin, "%lld", &p) != 1) {
            freeM(A);
            fclose(fin);
            fclose(fout);
            fprintf(stderr, "Error: invalid power\n");
            return 1;
        }
        Matrix *R = powM(A, p);
        freeM(A);
        if (!R) {
            fclose(fin);
            fclose(fout);
            fprintf(stderr, "Error: power failed\n");
            return 1;
        }
        writeM(fout, R);
        freeM(R);
        fclose(fin);
        fclose(fout);
        return 0;
    } else if (op == '|') {
        double d = detM(A);
        freeM(A);
        fclose(fin);
        fprintf(fout, "%g\n", d);
        fclose(fout);
        return 0;
    } else {
        freeM(A);
        fclose(fin);
        fclose(fout);
        fprintf(stderr, "Error: unknown operator %c\n", op);
        return 1;
    }
}