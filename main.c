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
static void freeM(Matrix *M){ if(M){ free(M->a); free(M);} }

static Matrix *readM(FILE *f){
    int r,c;
    if(fscanf(f,"%d%d",&r,&c)!=2) return NULL;
    Matrix *M=allocM(r,c);
    if(!M) return NULL;
    for(int i=0;i<r*c;i++)
        if(fscanf(f,"%lf",&M->a[i])!=1){freeM(M);return NULL;}
    return M;
}
static void writeM(FILE *f,const Matrix*M){
    fprintf(f,"%d %d\n",M->r,M->c);
    for(int i=0;i<M->r;i++){
        for(int j=0;j<M->c;j++)
            fprintf(f,"%.6f%s",M->a[i*M->c+j],j+1==M->c?"":" ");
        fputc('\n',f);
    }
}

static Matrix* sumM(const Matrix*A,const Matrix*B){
    if(!A||!B||A->r!=B->r||A->c!=B->c) return NULL;
    Matrix*R=allocM(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++)R->a[i]=A->a[i]+B->a[i];
    return R;
}
static Matrix* subM(const Matrix*A,const Matrix*B){
    if(!A||!B||A->r!=B->r||A->c!=B->c) return NULL;
    Matrix*R=allocM(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++)R->a[i]=A->a[i]-B->a[i];
    return R;
}
static Matrix* mulM(const Matrix*A,const Matrix*B){
    if(!A||!B||A->c!=B->r) return NULL;
    Matrix*R=allocM(A->r,B->c);
    for(int i=0;i<A->r;i++)
        for(int k=0;k<A->c;k++)
            for(int j=0;j<B->c;j++)
                R->a[i*R->c+j]+=A->a[i*A->c+k]*B->a[k*B->c+j];
    return R;
}
static double detM(const Matrix*A){
    if(!A||A->r!=A->c) return NAN;
    int n=A->r; Matrix*M=allocM(n,n);
    memcpy(M->a,A->a,sizeof(double)*n*n);
    double det=1;
    for(int i=0;i<n;i++){
        int piv=i;
        for(int r=i;r<n;r++)
            if(fabs(M->a[r*n+i])>fabs(M->a[piv*n+i])) piv=r;
        if(fabs(M->a[piv*n+i])<1e-12){det=0;break;}
        if(piv!=i){
            for(int j=0;j<n;j++){
                double t=M->a[i*n+j];
                M->a[i*n+j]=M->a[piv*n+j];
                M->a[piv*n+j]=t;
            }
            det=-det;
        }
        det*=M->a[i*n+i];
        double div=M->a[i*n+i];
        for(int r=i+1;r<n;r++){
            double f=M->a[r*n+i]/div;
            for(int j=i;j<n;j++) M->a[r*n+j]-=f*M->a[i*n+j];
        }
    }
    freeM(M); return det;
}
static Matrix*identity(int n){
    Matrix*I=allocM(n,n);
    for(int i=0;i<n;i++)I->a[i*n+i]=1;
    return I;
}
static Matrix*powM(const Matrix*A,long long p){
    if(!A||A->r!=A->c||p<0)return NULL;
    if(p==0)return identity(A->r);
    Matrix*R=identity(A->r);
    Matrix*B=allocM(A->r,A->c);
    memcpy(B->a,A->a,sizeof(double)*A->r*A->c);
    while(p>0){
        if(p&1){Matrix*t=mulM(R,B);freeM(R);R=t;}
        p>>=1;
        if(p){Matrix*t=mulM(B,B);freeM(B);B=t;}
    }
    freeM(B);return R;
}

/* ITMO-compatible command-line interface */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <op> <in1> [in2] <out>\n", argv[0]);
        return 1;
    }

    const char *op = argv[1];
    const char *in1 = argv[2];
    const char *in2 = (argc == 5) ? argv[3] : NULL;
    const char *outf = (argc == 5) ? argv[4] : argv[3];

    FILE *f1 = fopen(in1, "r");
    if (!f1) { fprintf(stderr, "Error: cannot open %s\n", in1); return 1; }

    FILE *fout = fopen(outf, "w");
    if (!fout) { fclose(f1); fprintf(stderr, "Error: cannot open %s\n", outf); return 1; }

    if (!strcmp(op, "det")) {
        Matrix *A = readM(f1);
        fclose(f1);
        if (!A) { fprintf(stderr, "Error: invalid matrix\n"); return 1; }
        fprintf(fout, "%.6f\n", detM(A));
        freeM(A);
        fclose(fout);
        return 0;
    }

    FILE *f2 = NULL;
    if (!strcmp(op, "sum") || !strcmp(op, "sub") || !strcmp(op, "mul")) {
        if (!in2) { fprintf(stderr, "Error: missing second matrix\n"); fclose(f1); fclose(fout); return 1; }
        f2 = fopen(in2, "r");
        if (!f2) { fprintf(stderr, "Error: cannot open %s\n", in2); fclose(f1); fclose(fout); return 1; }
        Matrix *A = readM(f1), *B = readM(f2);
        fclose(f1); fclose(f2);
        if (!A || !B) { fprintf(stderr, "Error: invalid matrix\n"); freeM(A); freeM(B); fclose(fout); return 1; }
        Matrix *R = !strcmp(op, "sum") ? sumM(A,B) :
                    !strcmp(op, "sub") ? subM(A,B) : mulM(A,B);
        writeM(fout, R);
        freeM(A); freeM(B); freeM(R);
        fclose(fout);
        return 0;
    }

    if (!strcmp(op, "pow")) {
        long long p;
        if (!in2) { fprintf(stderr, "Error: missing power file\n"); fclose(f1); fclose(fout); return 1; }
        FILE *fexp = fopen(in2, "r");
        if (!fexp) { fprintf(stderr, "Error: cannot open %s\n", in2); fclose(f1); fclose(fout); return 1; }
        Matrix *A = readM(f1);
        if (!A || fscanf(fexp, "%lld", &p)!=1) { fprintf(stderr, "Error: invalid input\n"); freeM(A); fclose(f1); fclose(fexp); fclose(fout); return 1; }
        fclose(f1); fclose(fexp);
        Matrix *R = powM(A,p);
        writeM(fout,R);
        freeM(A); freeM(R);
        fclose(fout);
        return 0;
    }

    fprintf(stderr, "Error: unknown op %s\n", op);
    fclose(f1); fclose(fout);
    return 1;
}
