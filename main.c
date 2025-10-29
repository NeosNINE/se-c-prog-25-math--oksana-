#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int rows, cols;
    float **data;
} Matrix;

Matrix alloc_matrix(int r, int c) {
    Matrix m;
    m.rows = r;
    m.cols = c;
    m.data = malloc(r * sizeof(float *));
    for (int i = 0; i < r; ++i)
        m.data[i] = malloc(c * sizeof(float));
    return m;
}

void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; ++i)
        free(m->data[i]);
    free(m->data);
    m->rows = m->cols = 0;
    m->data = NULL;
}

Matrix read_matrix(FILE *f, int r, int c) {
    Matrix m = alloc_matrix(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            fscanf(f, "%f", &m.data[i][j]);
    return m;
}

void print_matrix(Matrix m, FILE *f) {
    fprintf(f, "%d %d\n", m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            fprintf(f, "%g", m.data[i][j]);
            if (j < m.cols - 1)
                fprintf(f, " ");
        }
        fprintf(f, "\n");
    }
}

Matrix add_matrix(Matrix a, Matrix b) {
    Matrix r = alloc_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            r.data[i][j] = a.data[i][j] + b.data[i][j];
    return r;
}

Matrix sub_matrix(Matrix a, Matrix b) {
    Matrix r = alloc_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            r.data[i][j] = a.data[i][j] - b.data[i][j];
    return r;
}

Matrix mul_matrix(Matrix a, Matrix b) {
    Matrix r = alloc_matrix(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < b.cols; ++j) {
            r.data[i][j] = 0.0f;
            for (int k = 0; k < a.cols; ++k)
                r.data[i][j] += a.data[i][k] * b.data[k][j];
        }
    return r;
}

float det(Matrix m) {
    if (m.rows != m.cols)
        return NAN;

    if (m.rows == 1)
        return m.data[0][0];

    if (m.rows == 2)
        return m.data[0][0]*m.data[1][1] - m.data[0][1]*m.data[1][0];

    float d = 0.0f;
    for (int p = 0; p < m.cols; ++p) {
        Matrix sub = alloc_matrix(m.rows - 1, m.cols - 1);
        for (int i = 1; i < m.rows; ++i) {
            int cj = 0;
            for (int j = 0; j < m.cols; ++j) {
                if (j == p) continue;
                sub.data[i - 1][cj++] = m.data[i][j];
            }
        }
        float sign = (p % 2 == 0) ? 1.0f : -1.0f;
        d += sign * m.data[0][p] * det(sub);
        free_matrix(&sub);
    }
    return d;
}

Matrix identity_matrix(int n) {
    Matrix id = alloc_matrix(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            id.data[i][j] = (i == j) ? 1.0f : 0.0f;
    return id;
}

Matrix power_matrix(Matrix a, int n) {
    if (a.rows != a.cols)
        return alloc_matrix(0, 0);

    if (n == 0)
        return identity_matrix(a.rows);
    if (n == 1) {
        Matrix c = alloc_matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; ++i)
            for (int j = 0; j < a.cols; ++j)
                c.data[i][j] = a.data[i][j];
        return c;
    }

    Matrix res = identity_matrix(a.rows);
    for (int i = 0; i < n; ++i) {
        Matrix tmp = mul_matrix(res, a);
        free_matrix(&res);
        res = tmp;
    }
    return res;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    if (!fin || !fout) {
        fprintf(stderr, "File error.\n");
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fprintf(stderr, "Invalid operator.\n");
        return 1;
    }

    int ar, ac;
    if (fscanf(fin, "%d %d", &ar, &ac) != 2) {
        fprintf(stderr, "Invalid size.\n");
        return 1;
    }

    Matrix A = read_matrix(fin, ar, ac);
    Matrix B;
    Matrix R;
    int power_value = 0;

    // Determinant
    if (op == '|') {
        if (ar != ac) {
            fprintf(fout, "no solution\n");
        } else {
            float d = det(A);
            fprintf(fout, "%g\n", d);
        }
        free_matrix(&A);
        fclose(fin);
        fclose(fout);
        return 0;
    }

    // Power
    if (op == '^') {
        if (fscanf(fin, "%d", &power_value) != 1) {
            fprintf(stderr, "Missing exponent.\n");
            return 1;
        }
        if (ar != ac) {
            fprintf(fout, "no solution\n");
        } else {
            R = power_matrix(A, power_value);
            print_matrix(R, fout);
            free_matrix(&R);
        }
        free_matrix(&A);
        fclose(fin);
        fclose(fout);
        return 0;
    }

    // Second operand for + - *
    int br, bc;
    if (fscanf(fin, "%d %d", &br, &bc) != 2) {
        fprintf(stderr, "Missing second matrix size.\n");
        return 1;
    }
    B = read_matrix(fin, br, bc);

    switch (op) {
        case '+':
            if (ar != br || ac != bc) {
                fprintf(fout, "no solution\n");
            } else {
                R = add_matrix(A, B);
                print_matrix(R, fout);
                free_matrix(&R);
            }
            break;

        case '-':
            if (ar != br || ac != bc) {
                fprintf(fout, "no solution\n");
            } else {
                R = sub_matrix(A, B);
                print_matrix(R, fout);
                free_matrix(&R);
            }
            break;

        case '*':
            if (ac != br) {
                fprintf(fout, "no solution\n");
            } else {
                R = mul_matrix(A, B);
                print_matrix(R, fout);
                free_matrix(&R);
            }
            break;

        default:
            fprintf(stderr, "Unknown operator.\n");
            break;
    }

    free_matrix(&A);
    free_matrix(&B);
    fclose(fin);
    fclose(fout);
    return 0;
}
