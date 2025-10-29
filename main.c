#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ====== Структура матрицы ====== */
typedef struct {
    int r, c;
    double *a;
} Matrix;

/* ====== Память ====== */
static Matrix* alloc_matrix(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;
    M->r = r; M->c = c;
    M->a = calloc((size_t)r * (size_t)c, sizeof(double));
    if (!M->a) { free(M); return NULL; }
    return M;
}

static void free_matrix(Matrix *M) {
    if (M) { free(M->a); free(M); }
}

/* ====== Доступ ====== */
static inline double get(const Matrix *M, int i, int j) { return M->a[i * M->c + j]; }
static inline void set(Matrix *M, int i, int j, double v) { M->a[i * M->c + j] = v; }

/* ====== Ввод/вывод ====== */
static int read_matrix(FILE *f, Matrix **out) {
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) return -1;
    Matrix *M = alloc_matrix(r, c);
    if (!M) return -2;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            if (fscanf(f, "%lf", &M->a[i*M->c+j]) != 1) {
                free_matrix(M);
                return -3;
            }
    *out = M;
    return 0;
}

static int write_matrix(FILE *f, const Matrix *M) {
    fprintf(f, "%d %d\n", M->r, M->c);
    for (int i = 0; i < M->r; ++i) {
        for (int j = 0; j < M->c; ++j)
            fprintf(f, "%.6f%s", get(M,i,j), (j+1==M->c)?"":" ");
        fputc('\n', f);
    }
    return 0;
}

/* ====== Операции ====== */
static Matrix* sum_matrix(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r!=B->r || A->c!=B->c) return NULL;
    Matrix *R = alloc_matrix(A->r, A->c);
    for (int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]+B->a[i];
    return R;
}

static Matrix* sub_matrix(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r!=B->r || A->c!=B->c) return NULL;
    Matrix *R = alloc_matrix(A->r, A->c);
    for (int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]-B->a[i];
    return R;
}

static Matrix* mul_matrix(const Matrix *A, const Matrix *B) {
    if (!A||!B||A->c!=B->r) return NULL;
    Matrix *R = alloc_matrix(A->r, B->c);
    for (int i=0;i<A->r;i++)
        for (int k=0;k<A->c;k++)
            for (int j=0;j<B->c;j++)
                R->a[i*R->c+j]+=A->a[i*A->c+k]*B->a[k*B->c+j];
    return R;
}

static Matrix* identity(int n) {
    Matrix *I = alloc_matrix(n,n);
    for (int i=0;i<n;i++) set(I,i,i,1.0);
    return I;
}

static double det_matrix(const Matrix *A) {
    if (!A||A->r!=A->c) return NAN;
    int n=A->r;
    Matrix *M=alloc_matrix(n,n);
    memcpy(M->a,A->a,sizeof(double)*n*n);
    double det=1.0; int sign=1;
    for(int i=0;i<n;i++){
        int piv=i; double mx=fabs(get(M,i,i));
        for(int r=i+1;r<n;r++){
            double v=fabs(get(M,r,i));
            if(v>mx){mx=v;piv=r;}
        }
        if(mx<1e-15){det=0.0;goto done;}
        if(piv!=i){
            for(int j=0;j<n;j++){
                double t=get(M,i,j);
                set(M,i,j,get(M,piv,j));
                set(M,piv,j,t);
            }
            sign=-sign;
        }
        double d=get(M,i,i);
        det*=d;
        for(int r=i+1;r<n;r++){
            double f=get(M,r,i)/d;
            for(int j=i;j<n;j++)
                set(M,r,j,get(M,r,j)-f*get(M,i,j));
        }
    }
done:
    det*=sign; free_matrix(M); return det;
}

static Matrix* pow_matrix(const Matrix *A,long long p){
    if(!A||A->r!=A->c||p<0) return NULL;
    int n=A->r;
    Matrix *res=identity(n);
    Matrix *base=alloc_matrix(n,n);
    memcpy(base->a,A->a,sizeof(double)*n*n);
    while(p>0){
        if(p&1){
            Matrix*t=mul_matrix(res,base);
            free_matrix(res);
            res=t;
        }
        p>>=1;
        if(p){
            Matrix*t=mul_matrix(base,base);
            free_matrix(base);
            base=t;
        }
    }
    free_matrix(base);
    return res;
}

/* ====== CLI ====== */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: matrix <op> <input1> [input2] <output>\n");
        return 1;
    }

    const char *op = NULL;
    const char *in1 = NULL;
    const char *in2 = NULL;
    const char *out = NULL;

    // если операция указана явно
    if (!strcmp(argv[1], "sum") || !strcmp(argv[1], "sub") || !strcmp(argv[1], "mul") ||
        !strcmp(argv[1], "pow") || !strcmp(argv[1], "det")) {
        op  = argv[1];
        in1 = argv[2];
        if (argc == 5) { in2 = argv[3]; out = argv[4]; }
        else if (argc == 4) { out = argv[3]; }
        else { fprintf(stderr, "Invalid arguments.\n"); return 1; }
    } else {
        // без операции (det по умолчанию)
        op  = "det";
        if (argc == 3) { in1 = argv[1]; out = argv[2]; }
        else { fprintf(stderr, "Invalid arguments.\n"); return 1; }
    }

    Matrix *A=NULL,*B=NULL,*R=NULL;
    FILE *f1=fopen(in1,"r");
    if(!f1){fprintf(stderr,"Error: cannot open input file '%s'\n",in1);return 1;}
    if(read_matrix(f1,&A)){fclose(f1);fprintf(stderr,"Error: invalid matrix format in '%s'\n",in1);return 1;}
    fclose(f1);

    if (in2 && strcmp(op,"pow")) {
        FILE*f2=fopen(in2,"r");
        if(!f2){free_matrix(A);fprintf(stderr,"Error: cannot open second input '%s'\n",in2);return 1;}
        if(read_matrix(f2,&B)){fclose(f2);free_matrix(A);fprintf(stderr,"Error: invalid matrix format in '%s'\n",in2);return 1;}
        fclose(f2);
    }

    FILE *fo=fopen(out,"w");
    if(!fo){fprintf(stderr,"Error: cannot open output file '%s'\n",out);free_matrix(A);free_matrix(B);return 1;}

    int rc=0;
    if(!strcmp(op,"sum")){
        R=sum_matrix(A,B);
        if(!R){fprintf(stderr,"Error: invalid matrix sizes for sum.\n");rc=1;}
        else write_matrix(fo,R);
    }
    else if(!strcmp(op,"sub")){
        R=sub_matrix(A,B);
        if(!R){fprintf(stderr,"Error: invalid matrix sizes for sub.\n");rc=1;}
        else write_matrix(fo,R);
    }
    else if(!strcmp(op,"mul")){
        R=mul_matrix(A,B);
        if(!R){fprintf(stderr,"Error: invalid matrix sizes for mul.\n");rc=1;}
        else write_matrix(fo,R);
    }
    else if(!strcmp(op,"pow")){
        FILE *pf=fopen(in2,"r");
        long long p=0;
        if(!pf||fscanf(pf,"%lld",&p)!=1){
            fprintf(stderr,"Error: cannot read power value.\n");
            rc=1;
        } else {
            fclose(pf);
            R=pow_matrix(A,p);
            if(!R){fprintf(stderr,"Error: invalid power operation.\n");rc=1;}
            else write_matrix(fo,R);
        }
    }
    else if(!strcmp(op,"det")){
        double d=det_matrix(A);
        if(isnan(d)){fprintf(stderr,"Error: invalid determinant operation.\n");rc=1;}
        else fprintf(fo,"%.6f\n",d);
    }
    else {
        fprintf(stderr,"Error: unknown operation '%s'\n",op);
        rc=1;
    }

    fclose(fo);
    free_matrix(A);
    free_matrix(B);
    free_matrix(R);
    return rc;
}
