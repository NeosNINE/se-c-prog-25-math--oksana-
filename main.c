#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float **create_matrix(int r, int c) {
    float **m = malloc(r * sizeof(float *));
    if (!m) return NULL;
    for (int i = 0; i < r; ++i) {
        m[i] = malloc(c * sizeof(float));
        if (!m[i]) return NULL;
    }
    return m;
}

static void free_matrix(float **m, int r) {
    if (!m) return;
    for (int i = 0; i < r; ++i)
        free(m[i]);
    free(m);
}

static int read_matrix(FILE *f, float **m, int r, int c) {
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            if (fscanf(f, "%f", &m[i][j]) != 1)
                return 0;
    return 1;
}

static void print_matrix(FILE *f, float **m, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j)
            fprintf(f, "%g ", m[i][j]);
        fprintf(f, "\n");
    }
}

static float **identity_matrix(int n) {
    float **I = create_matrix(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            I[i][j] = (i == j) ? 1.0f : 0.0f;
    return I;
}

static float **copy_matrix(float **A, int r, int c) {
    float **B = create_matrix(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            B[i][j] = A[i][j];
    return B;
}

static float determinant(float **A, int n) {
    float **M = copy_matrix(A, n, n);
    float det = 1.0f;
    int swaps = 0;

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        float max_val = fabsf(M[i][i]);
        for (int k = i + 1; k < n; ++k)
            if (fabsf(M[k][i]) > max_val) {
                max_val = fabsf(M[k][i]);
                pivot = k;
            }

        if (fabsf(M[pivot][i]) < 1e-9f) {
            free_matrix(M, n);
            return 0.0f;
        }

        if (pivot != i) {
            float *tmp = M[i];
            M[i] = M[pivot];
            M[pivot] = tmp;
            swaps++;
        }

        det *= M[i][i];
        for (int k = i + 1; k < n; ++k) {
            float factor = M[k][i] / M[i][i];
            for (int j = i; j < n; ++j)
                M[k][j] -= factor * M[i][j];
        }
    }

    free_matrix(M, n);
    if (swaps % 2) det = -det;
    return det;
}

static float **add_matrices(float **A, float **B, int r, int c) {
    float **R = create_matrix(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            R[i][j] = A[i][j] + B[i][j];
    return R;
}

static float **sub_matrices(float **A, float **B, int r, int c) {
    float **R = create_matrix(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            R[i][j] = A[i][j] - B[i][j];
    return R;
}

static float **mul_matrices(float **A, float **B, int r1, int c1, int c2) {
    float **R = create_matrix(r1, c2);
    for (int i = 0; i < r1; ++i)
        for (int j = 0; j < c2; ++j) {
            R[i][j] = 0;
            for (int k = 0; k < c1; ++k)
                R[i][j] += A[i][k] * B[k][j];
        }
    return R;
}

static float **pow_matrix(float **A, int n, int p) {
    if (p < 0) return NULL;
    if (p == 0) return identity_matrix(n);
    if (p == 1) return copy_matrix(A, n, n);

    float **R = identity_matrix(n);
    float **B = copy_matrix(A, n, n);
    while (p > 0) {
        if (p % 2 == 1) {
            float **tmp = mul_matrices(R, B, n, n, n);
            free_matrix(R, n);
            R = tmp;
        }
        float **tmp = mul_matrices(B, B, n, n, n);
        free_matrix(B, n);
        B = tmp;
        p /= 2;
    }
    free_matrix(B, n);
    return R;
}

int main(int argc, char *argv[]) {
    if (argc != 3) return 0;

    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[2], "w");
    if (!in || !out) return 0;

    char op;
    if (fscanf(in, " %c", &op) != 1) {
        fprintf(out, "no solution\n");
        return 0;
    }

    int rA, cA;
    if (fscanf(in, "%d %d", &rA, &cA) != 2) {
        fprintf(out, "no solution\n");
        return 0;
    }

    float **A = create_matrix(rA, cA);
    if (!A || !read_matrix(in, A, rA, cA)) {
        fprintf(out, "no solution\n");
        return 0;
    }

    if (op == '|') {
        if (rA != cA) fprintf(out, "no solution\n");
        else fprintf(out, "%g\n", determinant(A, rA));

    } else if (op == '^') {
        int p;
        if (fscanf(in, "%d", &p) != 1 || rA != cA || p < 0) {
            fprintf(out, "no solution\n");
        } else {
            float **R = pow_matrix(A, rA, p);
            if (!R) fprintf(out, "no solution\n");
            else {
                print_matrix(out, R, rA, rA);
                free_matrix(R, rA);
            }
        }

    } else if (op == '+' || op == '-' || op == '*') {
        int rB, cB;
        if (fscanf(in, "%d %d", &rB, &cB) != 2) {
            fprintf(out, "no solution\n");
            return 0;
        }

        float **B = create_matrix(rB, cB);
        if (!B || !read_matrix(in, B, rB, cB)) {
            fprintf(out, "no solution\n");
            return 0;
        }

        float **R = NULL;
        if (op == '+' && rA == rB && cA == cB)
            R = add_matrices(A, B, rA, cA);
        else if (op == '-' && rA == rB && cA == cB)
            R = sub_matrices(A, B, rA, cA);
        else if (op == '*' && cA == rB)
            R = mul_matrices(A, B, rA, cA, cB);

        if (R) {
            print_matrix(out, R, (op == '*') ? rA : rA, (op == '*') ? cB : cA);
            free_matrix(R, rA);
        } else fprintf(out, "no solution\n");

        free_matrix(B, rB);
    } else {
        fprintf(out, "no solution\n");
    }

    free_matrix(A, rA);
    fclose(in);
    fclose(out);
    return 0;
}
