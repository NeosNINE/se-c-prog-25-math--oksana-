#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// === Matrix memory management ===
float **create_matrix(int rows, int cols) {
    float **m = malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
        m[i] = malloc(cols * sizeof(float));
    return m;
}

void free_matrix(float **m, int rows) {
    for (int i = 0; i < rows; i++)
        free(m[i]);
    free(m);
}

// === IO ===
void read_matrix(FILE *f, float **m, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (fscanf(f, "%f", &m[i][j]) != 1)
                exit(1);
}

void print_matrix(FILE *f, float **m, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float val = m[i][j];
            if (fabsf(val) < 1e-6f) val = 0.0f;
            fprintf(f, "%g", val);
            if (j < c - 1) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }
}

// === Helpers ===
float **copy_matrix(float **src, int r, int c) {
    float **dst = create_matrix(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            dst[i][j] = src[i][j];
    return dst;
}

float **identity_matrix(int n) {
    float **m = create_matrix(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = (i == j) ? 1.0f : 0.0f;
    return m;
}

// === Core operations ===
float calculate_determinant(float **a, int n) {
    float **m = copy_matrix(a, n, n);
    float det = 1.0f;
    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int k = i + 1; k < n; k++)
            if (fabsf(m[k][i]) > fabsf(m[pivot][i]))
                pivot = k;
        if (fabsf(m[pivot][i]) < 1e-10f) {
            free_matrix(m, n);
            return 0.0f;
        }
        if (pivot != i) {
            float *tmp = m[i];
            m[i] = m[pivot];
            m[pivot] = tmp;
            det = -det;
        }
        det *= m[i][i];
        for (int k = i + 1; k < n; k++) {
            float factor = m[k][i] / m[i][i];
            for (int j = i; j < n; j++)
                m[k][j] -= factor * m[i][j];
        }
    }
    free_matrix(m, n);
    return det;
}

float **add_matrices(float **A, float **B, int r, int c) {
    float **R = create_matrix(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            R[i][j] = A[i][j] + B[i][j];
    return R;
}

float **sub_matrices(float **A, float **B, int r, int c) {
    float **R = create_matrix(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            R[i][j] = A[i][j] - B[i][j];
    return R;
}

float **mul_matrices(float **A, float **B, int rA, int cA, int cB) {
    float **R = create_matrix(rA, cB);
    for (int i = 0; i < rA; i++)
        for (int j = 0; j < cB; j++) {
            R[i][j] = 0;
            for (int k = 0; k < cA; k++)
                R[i][j] += A[i][k] * B[k][j];
        }
    return R;
}

float **pow_matrix(float **A, int n, int p) {
    if (p == 0) return identity_matrix(n);
    if (p == 1) return copy_matrix(A, n, n);
    float **res = identity_matrix(n);
    float **base = copy_matrix(A, n, n);
    int power = p;
    while (power > 0) {
        if (power % 2) {
            float **tmp = mul_matrices(res, base, n, n, n);
            free_matrix(res, n);
            res = tmp;
        }
        float **tmp = mul_matrices(base, base, n, n, n);
        free_matrix(base, n);
        base = tmp;
        power /= 2;
    }
    free_matrix(base, n);
    return res;
}

// === Main ===
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "no solution\n");
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    if (!fin || !fout) {
        fprintf(stderr, "no solution\n");
        return 1;
    }

    char op;
    if (fscanf(fin, " %c", &op) != 1) {
        fprintf(stderr, "no solution\n");
        return 1;
    }

    int rA, cA;
    if (fscanf(fin, "%d %d", &rA, &cA) != 2) {
        fprintf(stderr, "no solution\n");
        return 1;
    }

    float **A = create_matrix(rA, cA);
    read_matrix(fin, A, rA, cA);

    if (op == '|') {
        if (rA != cA) {
            fprintf(fout, "no solution\n");
        } else {
            float det = calculate_determinant(A, rA);
            fprintf(fout, "%g\n", det);
        }
    }

    else if (op == '^') {
        int power;
        if (fscanf(fin, "%d", &power) != 1 || power < 0 || rA != cA) {
            fprintf(fout, "no solution\n");
        } else {
            float **res = pow_matrix(A, rA, power);
            print_matrix(fout, res, rA, rA);
            free_matrix(res, rA);
        }
    }

    else if (op == '+' || op == '-' || op == '*') {
        int rB, cB;
        if (fscanf(fin, "%d %d", &rB, &cB) != 2) {
            fprintf(fout, "no solution\n");
            free_matrix(A, rA);
            fclose(fin);
            fclose(fout);
            return 0;
        }

        float **B = create_matrix(rB, cB);
        read_matrix(fin, B, rB, cB);

        float **res = NULL;
        if (op == '+') {
            if (rA == rB && cA == cB)
                res = add_matrices(A, B, rA, cA);
            else
                fprintf(fout, "no solution\n");
        } else if (op == '-') {
            if (rA == rB && cA == cB)
                res = sub_matrices(A, B, rA, cA);
            else
                fprintf(fout, "no solution\n");
        } else if (op == '*') {
            if (cA == rB)
                res = mul_matrices(A, B, rA, cA, cB);
            else
                fprintf(fout, "no solution\n");
        }

        if (res) {
            print_matrix(fout, res, rA, (op == '*' ? cB : cA));
            free_matrix(res, rA);
        }

        free_matrix(B, rB);
    }

    else {
        fprintf(fout, "no solution\n");
    }

    free_matrix(A, rA);
    fclose(fin);
    fclose(fout);
    return 0;
}
