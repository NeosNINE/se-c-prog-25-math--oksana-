int main(void) {
    char op[8];
    if (scanf("%7s", op) != 1) {
        fprintf(stderr, "Error: no operator\n");
        return 1;
    }

    if (!strcmp(op, "det")) {
        Matrix *A = readM(stdin);
        if (!A) { fprintf(stderr, "Error: invalid matrix\n"); return 1; }
        printf("%.6f\n", detM(A));
        freeM(A);
        return 0;
    }

    if (!strcmp(op, "sum")) {
        Matrix *A = readM(stdin), *B = readM(stdin);
        if (!A || !B) { fprintf(stderr, "Error: invalid matrix\n"); freeM(A); freeM(B); return 1; }
        Matrix *R = sumM(A,B);
        writeM(stdout, R);
        freeM(A); freeM(B); freeM(R);
        return 0;
    }

    if (!strcmp(op, "sub")) {
        Matrix *A = readM(stdin), *B = readM(stdin);
        if (!A || !B) { fprintf(stderr, "Error: invalid matrix\n"); freeM(A); freeM(B); return 1; }
        Matrix *R = subM(A,B);
        writeM(stdout, R);
        freeM(A); freeM(B); freeM(R);
        return 0;
    }

    if (!strcmp(op, "mul")) {
        Matrix *A = readM(stdin), *B = readM(stdin);
        if (!A || !B) { fprintf(stderr, "Error: invalid matrix\n"); freeM(A); freeM(B); return 1; }
        Matrix *R = mulM(A,B);
        writeM(stdout, R);
        freeM(A); freeM(B); freeM(R);
        return 0;
    }

    if (!strcmp(op, "pow")) {
        Matrix *A = readM(stdin);
        long long p;
        if (!A || scanf("%lld", &p) != 1) { fprintf(stderr, "Error: invalid input\n"); freeM(A); return 1; }
        Matrix *R = powM(A, p);
        writeM(stdout, R);
        freeM(A); freeM(R);
        return 0;
    }

    fprintf(stderr, "Error: unknown op\n");
    return 1;
}
