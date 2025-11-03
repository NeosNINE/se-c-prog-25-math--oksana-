// main.c
// ЛР-1: матричный калькулятор, C23 совместимо (gcc: -std=c2x -O2 -lm)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

/* ---------- утилиты матриц ---------- */

static Matrix alloc_matrix(int r, int c) {
    Matrix m = { r, c, NULL };
    if (r <= 0 || c <= 0) { m.rows = m.cols = 0; return m; }

    m.data = (float **)malloc((size_t)r * sizeof(float *));
    if (!m.data) { m.rows = m.cols = 0; return m; }

    for (int i = 0; i < r; ++i) {
        m.data[i] = (float *)malloc((size_t)c * sizeof(float));
        if (!m.data[i]) {
            for (int k = 0; k < i; ++k) free(m.data[k]);
            free(m.data);
            m.data = NULL;
            m.rows = m.cols = 0;
            return m;
        }
    }
    return m;
}

static void free_matrix(Matrix *m) {
    if (!m || !m->data) return;
    for (int i = 0; i < m->rows; ++i) free(m->data[i]);
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

static int read_matrix(FILE *f, Matrix *m, int r, int c) {
    *m = alloc_matrix(r, c);
    if (!m->data) return 0;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            if (fscanf(f, "%f", &m->data[i][j]) != 1) {
                free_matrix(m);
                return 0;
            }
        }
    }
    return 1;
}

static void print_matrix(FILE *f, const Matrix *m) {
    fprintf(f, "%d %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            fprintf(f, "%g", m->data[i][j]);
            if (j + 1 < m->cols) fputc(' ', f);
        }
        fputc('\n', f);
    }
}

static Matrix add_matrix(const Matrix *a, const Matrix *b) {
    Matrix r = alloc_matrix(a->rows, a->cols);
    if (!r.data) return r;
    for (int i = 0; i < a->rows; ++i)
        for (int j = 0; j < a->cols; ++j)
            r.data[i][j] = a->data[i][j] + b->data[i][j];
    return r;
}

static Matrix sub_matrix(const Matrix *a, const Matrix *b) {
    Matrix r = alloc_matrix(a->rows, a->cols);
    if (!r.data) return r;
    for (int i = 0; i < a->rows; ++i)
        for (int j = 0; j < a->cols; ++j)
            r.data[i][j] = a->data[i][j] - b->data[i][j];
    return r;
}

static Matrix mul_matrix(const Matrix *a, const Matrix *b) {
    Matrix r = alloc_matrix(a->rows, b->cols);
    if (!r.data) return r;
    for (int i = 0; i < a->rows; ++i) {
        for (int j = 0; j < b->cols; ++j) {
            float s = 0.0f;
            for (int k = 0; k < a->cols; ++k)
                s += a->data[i][k] * b->data[k][j];
            r.data[i][j] = s;
        }
    }
    return r;
}

static Matrix identity_matrix(int n) {
    Matrix I = alloc_matrix(n, n);
    if (!I.data) return I;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) I.data[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    return I;
}

static Matrix pow_matrix(const Matrix *a, int p) {
    // p — натуральное (по ТЗ). Обработаем p = 0 на всякий случай.
    if (a->rows != a->cols) return alloc_matrix(0, 0);
    int n = a->rows;

    if (p == 0) return identity_matrix(n);

    // Быстрое возведение в степень (binary exponentiation)
    Matrix base = alloc_matrix(n, n);
    if (!base.data) return base;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            base.data[i][j] = a->data[i][j];

    Matrix res = identity_matrix(n);
    if (!res.data) { free_matrix(&base); return res; }

    int e = p;
    while (e > 0) {
        if (e & 1) {
            Matrix tmp = mul_matrix(&res, &base);
            free_matrix(&res);
            res = tmp;
            if (!res.data) { free_matrix(&base); return res; }
        }
        e >>= 1;
        if (e > 0) {
            Matrix tmp = mul_matrix(&base, &base);
            free_matrix(&base);
            base = tmp;
            if (!base.data) { free_matrix(&res); return base; }
        }
    }
    free_matrix(&base);
    return res;
}

/* Детерминант: Гаусс с частичным выбором главного элемента */
static int approximately_zero(float x) {
    const float eps = 1e-6f;
    return fabsf(x) < eps;
}

static float det_matrix(const Matrix *a) {
    if (a->rows != a->cols) return 0.0f;

    int n = a->rows;

    // Копия, чтобы не портить исходную
    Matrix m = alloc_matrix(n, n);
    if (!m.data) return 0.0f;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m.data[i][j] = a->data[i][j];

    double det = 1.0;
    int sign = 1;

    for (int col = 0; col < n; ++col) {
        // поиск максимального по модулю элемента в текущем столбце
        int pivot = col;
        float mx = fabsf(m.data[col][col]);
        for (int i = col + 1; i < n; ++i) {
            float v = fabsf(m.data[i][col]);
            if (v > mx) { mx = v; pivot = i; }
        }

        if (approximately_zero(mx)) { det = 0.0; sign = 1; break; }

        if (pivot != col) {
            float *tmp = m.data[pivot];
            m.data[pivot] = m.data[col];
            m.data[col] = tmp;
            sign = -sign;
        }

        float diag = m.data[col][col];
        det *= diag;

        // исключение вниз
        for (int i = col + 1; i < n; ++i) {
            float factor = m.data[i][col] / diag;
            if (approximately_zero(factor)) continue;
            m.data[i][col] = 0.0f;
            for (int j = col + 1; j < n; ++j)
                m.data[i][j] -= factor * m.data[col][j];
        }
    }

    free_matrix(&m);
    det *= sign;
    return (float)det;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    if (!fin || !fout) {
        fprintf(stderr, "File error.\n");
        if (fin) fclose(fin);
        if (fout) fclose(fout);
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fprintf(stderr, "Missing operator.\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    int ar, ac;
    if (fscanf(fin, "%d %d", &ar, &ac) != 2) {
        fprintf(stderr, "Missing first matrix size.\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    Matrix A = {0};
    if (!read_matrix(fin, &A, ar, ac)) {
        fprintf(stderr, "Failed to read first matrix.\n");
        fclose(fin);
        fclose(fout);
        return 1;
    }

    int exit_code = 0;

    if (op == '|') {
        if (A.rows != A.cols) {
            fprintf(fout, "no solution\n");
        } else {
            float d = det_matrix(&A);
            fprintf(fout, "%g\n", d);
        }
    } else if (op == '^') {
        int p;
        if (fscanf(fin, "%d", &p) != 1) {
            fprintf(stderr, "Missing power value.\n");
            free_matrix(&A);
            fclose(fin);
            fclose(fout);
            return 1;
        }
        if (A.rows != A.cols) {
            fprintf(fout, "no solution\n");
        } else {
            Matrix P = pow_matrix(&A, p);
            if (!P.data) { free_matrix(&A); fclose(fin); fclose(fout); return 1; }
            print_matrix(fout, &P);
            free_matrix(&P);
        }
    } else {
        int br, bc;
        if (fscanf(fin, "%d %d", &br, &bc) != 2) {
            fprintf(stderr, "Missing second matrix size.\n");
            free_matrix(&A);
            fclose(fin);
            fclose(fout);
            return 1;
        }
        Matrix B = {0};
        if (!read_matrix(fin, &B, br, bc)) {
            fprintf(stderr, "Failed to read second matrix.\n");
            free_matrix(&A);
            fclose(fin);
            fclose(fout);
            return 1;
        }

        switch (op) {
            case '+':
                if (A.rows != B.rows || A.cols != B.cols) {
                    fprintf(fout, "no solution\n");
                } else {
                    Matrix R = add_matrix(&A, &B);
                    print_matrix(fout, &R);
                    free_matrix(&R);
                }
                break;

            case '-':
                if (A.rows != B.rows || A.cols != B.cols) {
                    fprintf(fout, "no solution\n");
                } else {
                    Matrix R = sub_matrix(&A, &B);
                    print_matrix(fout, &R);
                    free_matrix(&R);
                }
                break;

            case '*':
                if (A.cols != B.rows) {
                    fprintf(fout, "no solution\n");
                } else {
                    Matrix R = mul_matrix(&A, &B);
                    print_matrix(fout, &R);
                    free_matrix(&R);
                }
                break;

            default:
                fprintf(stderr, "Unknown operator.\n");
                exit_code = 1;
        }

        free_matrix(&B);
    }

    free_matrix(&A);
    fclose(fin);
    fclose(fout);
    return exit_code;
}
