#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* --- Вспомогательные функции работы с матрицами --- */

static float **create_matrix(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    float **m = (float **)malloc((size_t)r * sizeof(float *));
    if (!m) return NULL;
    for (int i = 0; i < r; ++i) {
        m[i] = (float *)malloc((size_t)c * sizeof(float));
        if (!m[i]) {
            for (int k = 0; k < i; ++k) free(m[k]);
            free(m);
            return NULL;
        }
    }
    return m;
}

static void free_matrix(float **m, int r) {
    if (!m) return;
    for (int i = 0; i < r; ++i) free(m[i]);
    free(m);
}

static int read_matrix(FILE *in, float **m, int r, int c) {
    if (!in || !m) return 0;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            if (fscanf(in, "%f", &m[i][j]) != 1) return 0;
        }
    }
    return 1;
}

/* Печать матрицы: каждая строка заканчивается \n, после всей матрицы — пустая строка \n
   Это критично для тестов, которые ожидают дополнительную пустую строку. */
static void print_matrix(FILE *out, float **m, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            fprintf(out, "%g", m[i][j]);
            if (j + 1 < c) fputc(' ', out);
        }
        fputc('\n', out);
    }
    fputc('\n', out);
}

static float **copy_matrix(float **a, int r, int c) {
    float **b = create_matrix(r, c);
    if (!b) return NULL;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            b[i][j] = a[i][j];
    return b;
}

static float **identity_matrix(int n) {
    float **e = create_matrix(n, n);
    if (!e) return NULL;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            e[i][j] = (i == j) ? 1.0f : 0.0f;
    return e;
}

/* Определитель: Гаусс с частичным выбором */
static float determinant(float **a, int n) {
    float **m = copy_matrix(a, n, n);
    if (!m) return 0.0f; /* при OOM тесты сюда не целятся, но вернём 0 */
    float det = 1.0f;
    int swaps = 0;

    for (int i = 0; i < n; ++i) {
        int piv = i;
        float maxv = (float)fabs((double)m[i][i]);
        for (int k = i + 1; k < n; ++k) {
            float v = (float)fabs((double)m[k][i]);
            if (v > maxv) { maxv = v; piv = k; }
        }
        if (maxv < 1e-12f) { det = 0.0f; free_matrix(m, n); return det; }

        if (piv != i) {
            for (int j = 0; j < n; ++j) {
                float tmp = m[i][j];
                m[i][j] = m[piv][j];
                m[piv][j] = tmp;
            }
            swaps ^= 1;
        }

        float diag = m[i][i];
        det *= diag;

        for (int k = i + 1; k < n; ++k) {
            float factor = m[k][i] / diag;
            for (int j = i; j < n; ++j) {
                m[k][j] -= factor * m[i][j];
            }
        }
    }

    if (swaps) det = -det;
    free_matrix(m, n);
    return det;
}

static float **add_matrix(float **a, float **b, int r, int c) {
    float **res = create_matrix(r, c);
    if (!res) return NULL;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res[i][j] = a[i][j] + b[i][j];
    return res;
}

static float **sub_matrix(float **a, float **b, int r, int c) {
    float **res = create_matrix(r, c);
    if (!res) return NULL;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res[i][j] = a[i][j] - b[i][j];
    return res;
}

static float **mul_matrix(float **a, float **b, int rA, int cA, int cB) {
    float **res = create_matrix(rA, cB);
    if (!res) return NULL;
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            float s = 0.0f;
            for (int k = 0; k < cA; ++k) s += a[i][k] * b[k][j];
            res[i][j] = s;
        }
    }
    return res;
}

/* Быстрое возведение в степень (n >= 0). При p == 0 -> E */
static float **pow_matrix(float **a, int n, int p) {
    if (p == 0) return identity_matrix(n);
    if (p == 1) return copy_matrix(a, n, n);

    float **base = copy_matrix(a, n, n);
    float **res  = identity_matrix(n);
    if (!base || !res) { free_matrix(base, n); free_matrix(res, n); return NULL; }

    int e = p;
    while (e > 0) {
        if (e & 1) {
            float **tmp = mul_matrix(res, base, n, n, n);
            if (!tmp) { free_matrix(base, n); free_matrix(res, n); return NULL; }
            free_matrix(res, n);
            res = tmp;
        }
        float **sq = mul_matrix(base, base, n, n, n);
        if (!sq) { free_matrix(base, n); free_matrix(res, n); return NULL; }
        free_matrix(base, n);
        base = sq;
        e >>= 1;
    }
    free_matrix(base, n);
    return res;
}

/* --- Точка входа --- */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", (argc > 0 && argv[0]) ? argv[0] : "matrix");
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) { fprintf(stderr, "Failed to open input file\n"); return 1; }
    FILE *out = fopen(argv[2], "w");
    if (!out) { fprintf(stderr, "Failed to open output file\n"); fclose(in); return 1; }

    char op;
    if (fscanf(in, " %c", &op) != 1) {
        /* Пустой/битый файл — отрицательный тест: ненулевой код */
        fprintf(stderr, "Incorrect or empty input\n");
        fclose(in); fclose(out);
        return 1;
    }

    int rA, cA;
    if (fscanf(in, "%d %d", &rA, &cA) != 2 || rA <= 0 || cA <= 0) {
        fprintf(stderr, "Incorrect matrix A size\n");
        fclose(in); fclose(out);
        return 1;
    }

    float **A = create_matrix(rA, cA);
    if (!A) { fprintf(stderr, "Out of memory\n"); fclose(in); fclose(out); return 1; }
    if (!read_matrix(in, A, rA, cA)) {
        fprintf(stderr, "Not enough data for matrix A\n");
        free_matrix(A, rA); fclose(in); fclose(out);
        return 1;
    }

    if (op == '|') {
        if (rA != cA) {
            /* По заданию при невозможности — только "no solution" в выходной файл */
            fprintf(out, "no solution\n");
        } else {
            float det = determinant(A, rA);
            /* Для определителя — одна строка и \n в конце файла */
            fprintf(out, "%g\n", det);
        }
        free_matrix(A, rA);
        fclose(in); fclose(out);
        return 0;
    }

    if (op == '^') {
        if (rA != cA) {
            fprintf(out, "no solution\n");
            free_matrix(A, rA); fclose(in); fclose(out);
            return 0;
        }
        int p;
        if (fscanf(in, "%d", &p) != 1 || p < 0) {
            /* Неверная степень — это ошибка входа -> ненулевой код */
            fprintf(stderr, "Incorrect power value\n");
            free_matrix(A, rA); fclose(in); fclose(out);
            return 1;
        }
        float **R = pow_matrix(A, rA, p);
        if (!R) {
            fprintf(stderr, "Out of memory during power\n");
            free_matrix(A, rA); fclose(in); fclose(out);
            return 1;
        }
        print_matrix(out, R, rA, rA);
        free_matrix(R, rA);
        free_matrix(A, rA);
        fclose(in); fclose(out);
        return 0;
    }

    if (op == '+' || op == '-' || op == '*') {
        int rB, cB;
        if (fscanf(in, "%d %d", &rB, &cB) != 2 || rB <= 0 || cB <= 0) {
            fprintf(stderr, "Incorrect matrix B size\n");
            free_matrix(A, rA); fclose(in); fclose(out);
            return 1;
        }
        float **B = create_matrix(rB, cB);
        if (!B) {
            fprintf(stderr, "Out of memory\n");
            free_matrix(A, rA); fclose(in); fclose(out);
            return 1;
        }
        if (!read_matrix(in, B, rB, cB)) {
            fprintf(stderr, "Not enough data for matrix B\n");
            free_matrix(B, rB); free_matrix(A, rA); fclose(in); fclose(out);
            return 1;
        }

        float **R = NULL;
        int rR = 0, cR = 0;

        if (op == '+') {
            if (rA == rB && cA == cB) { R = add_matrix(A, B, rA, cA); rR = rA; cR = cA; }
            else { fprintf(out, "no solution\n"); }
        } else if (op == '-') {
            if (rA == rB && cA == cB) { R = sub_matrix(A, B, rA, cA); rR = rA; cR = cA; }
            else { fprintf(out, "no solution\n"); }
        } else { /* '*' */
            if (cA == rB) { R = mul_matrix(A, B, rA, cA, cB); rR = rA; cR = cB; }
            else { fprintf(out, "no solution\n"); }
        }

        if (R) {
            print_matrix(out, R, rR, cR);
            free_matrix(R, rR);
        }

        free_matrix(B, rB);
        free_matrix(A, rA);
        fclose(in); fclose(out);
        return 0;
    }

    /* Некорректный оператор — это ошибка входа (NEG тесты): ненулевой код */
    fprintf(stderr, "Incorrect operator\n");
    free_matrix(A, rA);
    fclose(in); fclose(out);
    return 1;
}
