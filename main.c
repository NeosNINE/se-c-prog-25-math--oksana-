#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ===== структура ===== */
typedef struct {
    int r, c;
    double *a;
} Matrix;

/* ===== память ===== */
static Matrix* alloc_matrix(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    Matrix *M = malloc(sizeof(Matrix));
    if (!M) return NULL;
    M->r = r; M->c = c;
    M->a = calloc((size_t)r * (size_t)c, sizeof(double));
    if (!M->a) { free(M); return NULL; }
    return M;
}
static void free_matrix(Matrix *M) { if (M) { free(M->a); free(M); } }

/* ===== чтение матрицы ===== */
static int read_matrix(FILE *f, Matrix **out) {
    if (!f) return -1;
    char buf[8192];
    double *vals = NULL;
    int rows = 0, cols = 0, capacity = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char *p = buf;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '\n') continue; // пустая строка
        double temp[1024]; int cnt = 0;
        while (sscanf(p, "%lf", &temp[cnt]) == 1) {
            cnt++;
            while (*p && !isspace((unsigned char)*p)) p++;
            while (isspace((unsigned char)*p)) p++;
        }
        if (cnt == 0) continue;
        if (cols == 0) cols = cnt;
        else if (cnt != cols) { free(vals); return -2; }
        if (rows * cols + cnt > capacity) {
            capacity = (capacity == 0 ? 1024 : capacity * 2);
            vals = realloc(vals, capacity * sizeof(double));
        }
        memcpy(vals + rows * cols, temp, cnt * sizeof(double));
        rows++;
    }
    if (rows == 0 || cols == 0) { free(vals); return -3; }
    Matrix *M = alloc_matrix(rows, cols);
    if (!M) { free(vals); return -4; }
    memcpy(M->a, vals, rows * cols * sizeof(double));
    free(vals);
    *out = M;
    return 0;
}

/* ===== запись ===== */
static void write_matrix(FILE *f, const Matrix *M) {
    fprintf(f, "%d %d\n", M->r, M->c);
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++)
            fprintf(f, "%.6f%s", M->a[i*M->c+j], j+1==M->c?"":" ");
        fputc('\n', f);
    }
}

/* ===== операции ===== */
static Matrix* sum_matrix(const Matrix*A,const Matrix*B){
    if(!A||!B||A->r!=B->r||A->c!=B->c) return NULL;
    Matrix*R=alloc_matrix(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]+B->a[i];
    return R;
}
static Matrix* sub_matrix(const Matrix*A,const Matrix*B){
    if(!A||!B||A->r!=B->r||A->c!=B->c) return NULL;
    Matrix*R=alloc_matrix(A->r,A->c);
    for(int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]-B->a[i];
    return R;
}
static Matrix* mul_matrix(const Matrix*A,const Matrix*B){
    if(!A||!B||A->c!=B->r) return NULL;
    Matrix*R=alloc_matrix(A->r,B->c);
    for(int i=0;i<A->r;i++)
        for(int k=0;k<A->c;k++)
            for(int j=0;j<B->c;j++)
                R->a[i*R->c+j]+=A->a[i*A->c+k]*B->a[k*B->c+j];
    return R;
}
static Matrix* identity(int n){
    Matrix*I=alloc_matrix(n,n);
    for(int i=0;i<n;i++) I->a[i*n+i]=1.0;
    return I;
}
static double det_matrix(const Matrix*A){
    if(!A||A->r!=A->c) return NAN;
    int n=A->r; Matrix*M=alloc_matrix(n,n);
    memcpy(M->a,A->a,sizeof(double)*n*n);
    double det=1.0; int sgn=1;
    for(int i=0;i<n;i++){
        int piv=i; double mx=fabs(M->a[i*n+i]);
        for(int r=i+1;r<n;r++){
            double v=fabs(M->a[r*n+i]);
            if(v>mx){mx=v;piv=r;}
        }
        if(mx<1e-15){det=0;goto done;}
        if(piv!=i){
            for(int j=0;j<n;j++){
                double t=M->a[i*n+j];
                M->a[i*n+j]=M->a[piv*n+j];
                M->a[piv*n+j]=t;
            }
            sgn=-sgn;
        }
        double d=M->a[i*n+i];
        det*=d;
        for(int r=i+1;r<n;r++){
            double f=M->a[r*n+i]/d;
            for(int j=i;j<n;j++)
                M->a[r*n+j]-=f*M->a[i*n+j];
        }
    }
done:
    free_matrix(M);
    return det*sgn;
}
static Matrix* pow_matrix(const Matrix*A,long long p){
    if(!A||A->r!=A->c||p<0)return NULL;
    int n=A->r; Matrix*res=identity(n);
    Matrix*base=alloc_matrix(n,n);
    memcpy(base->a,A->a,sizeof(double)*n*n);
    while(p>0){
        if(p&1){Matrix*t=mul_matrix(res,base);free_matrix(res);res=t;}
        p>>=1;
        if(p){Matrix*t=mul_matrix(base,base);free_matrix(base);base=t;}
    }
    free_matrix(base);
    return res;
}

/* ===== CLI ===== */
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: matrix <op> <input1> [input2] <output>\n");return 1;}
    const char*op=NULL;const char*in1=NULL;const char*in2=NULL;const char*out=NULL;
    if(!strcmp(argv[1],"sum")||!strcmp(argv[1],"sub")||!strcmp(argv[1],"mul")||
       !strcmp(argv[1],"pow")||!strcmp(argv[1],"det")){
        op=argv[1];in1=argv[2];
        if(argc==5){in2=argv[3];out=argv[4];}
        else if(argc==4){out=argv[3];}
        else{fprintf(stderr,"Invalid arguments.\n");return 1;}
    }else{
        op="det";
        if(argc==3){in1=argv[1];out=argv[2];}
        else{fprintf(stderr,"Invalid arguments.\n");return 1;}
    }

    Matrix*A=NULL,*B=NULL,*R=NULL;
    FILE*f1=fopen(in1,"r");
    if(!f1){fprintf(stderr,"Error: cannot open input '%s'\n",in1);return 1;}
    if(read_matrix(f1,&A)){fclose(f1);fprintf(stderr,"Error: invalid matrix in '%s'\n",in1);return 1;}
    fclose(f1);

    if(in2&&strcmp(op,"pow")){
        FILE*f2=fopen(in2,"r");
        if(!f2){fprintf(stderr,"Error: cannot open second '%s'\n",in2);free_matrix(A);return 1;}
        if(read_matrix(f2,&B)){fclose(f2);fprintf(stderr,"Error: invalid matrix in '%s'\n",in2);free_matrix(A);return 1;}
        fclose(f2);
    }

    FILE*fo=fopen(out,"w");
    if(!fo){fprintf(stderr,"Error: cannot open output '%s'\n",out);free_matrix(A);free_matrix(B);return 1;}

    int rc=0;
    if(!strcmp(op,"sum")){R=sum_matrix(A,B);if(!R){fprintf(stderr,"Error: size mismatch.\n");rc=1;}else write_matrix(fo,R);}
    else if(!strcmp(op,"sub")){R=sub_matrix(A,B);if(!R){fprintf(stderr,"Error: size mismatch.\n");rc=1;}else write_matrix(fo,R);}
    else if(!strcmp(op,"mul")){R=mul_matrix(A,B);if(!R){fprintf(stderr,"Error: size mismatch.\n");rc=1;}else write_matrix(fo,R);}
    else if(!strcmp(op,"pow")){
        FILE*pf=fopen(in2,"r");long long p;
        if(!pf||fscanf(pf,"%lld",&p)!=1){fprintf(stderr,"Error: bad power file.\n");rc=1;}
        else{fclose(pf);R=pow_matrix(A,p);if(!R){fprintf(stderr,"Error: power failed.\n");rc=1;}else write_matrix(fo,R);}
    }else if(!strcmp(op,"det")){
        double d=det_matrix(A);
        if(isnan(d)){fprintf(stderr,"Error: det failed.\n");rc=1;}else fprintf(fo,"%.6f\n",d);
    }else{fprintf(stderr,"Error: unknown op.\n");rc=1;}

    fclose(fo);free_matrix(A);free_matrix(B);free_matrix(R);
    return rc;
}
