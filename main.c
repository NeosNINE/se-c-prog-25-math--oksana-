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

static int readM(const char *path, Matrix **out) {
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    int r, c;
    if (fscanf(f, "%d%d", &r, &c) != 2) { fclose(f); return 2; }
    Matrix *M = allocM(r, c);
    if (!M) { fclose(f); return 3; }
    for (int i = 0; i < r * c; i++)
        if (fscanf(f, "%lf", &M->a[i]) != 1) { freeM(M); fclose(f); return 4; }
    fclose(f);
    *out = M;
    return 0;
}
static int writeM(const char *path, const Matrix *M) {
    FILE *f = fopen(path, "w");
    if (!f) return 1;
    fprintf(f, "%d %d\n", M->r, M->c);
    for (int i = 0; i < M->r; i++) {
        for (int j = 0; j < M->c; j++)
            fprintf(f, "%.6f%s", M->a[i*M->c+j], j+1==M->c?"":" ");
        fputc('\n', f);
    }
    fclose(f);
    return 0;
}

/* === operations === */
static Matrix *sumM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r!=B->r || A->c!=B->c) return NULL;
    Matrix *R = allocM(A->r,A->c);
    for (int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]+B->a[i];
    return R;
}
static Matrix *subM(const Matrix *A, const Matrix *B) {
    if (!A || !B || A->r!=B->r || A->c!=B->c) return NULL;
    Matrix *R = allocM(A->r,A->c);
    for (int i=0;i<A->r*A->c;i++) R->a[i]=A->a[i]-B->a[i];
    return R;
}
static Matrix *mulM(const Matrix *A, const Matrix *B) {
    if (!A||!B||A->c!=B->r) return NULL;
    Matrix *R=allocM(A->r,B->c);
    for (int i=0;i<A->r;i++)
        for (int k=0;k<A->c;k++)
            for (int j=0;j<B->c;j++)
                R->a[i*R->c+j]+=A->a[i*A->c+k]*B->a[k*B->c+j];
    return R;
}
static double detM(const Matrix *A) {
    if (!A||A->r!=A->c) return NAN;
    int n=A->r;
    Matrix *M=allocM(n,n);
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
    freeM(M);
    return det;
}
static Matrix *identity(int n) {
    Matrix *I=allocM(n,n);
    for(int i=0;i<n;i++) I->a[i*n+i]=1;
    return I;
}
static Matrix *powM(const Matrix *A,long long p) {
    if(!A||A->r!=A->c||p<0) return NULL;
    Matrix *R=identity(A->r);
    Matrix *B=allocM(A->r,A->c);
    memcpy(B->a,A->a,sizeof(double)*A->r*A->c);
    while(p>0){
        if(p&1){
            Matrix *tmp=mulM(R,B);
            freeM(R); R=tmp;
        }
        p>>=1;
        if(p){
            Matrix *tmp=mulM(B,B);
            freeM(B); B=tmp;
        }
    }
    freeM(B);
    return R;
}

int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"Usage: %s <op> <in1> [in2] <out>\n",argv[0]);return 1;}
    const char*op=argv[1];
    const char*in1=argv[2];
    const char*out=argv[argc-1];
    Matrix *A=NULL,*B=NULL,*R=NULL;
    if(readM(in1,&A)){fprintf(stderr,"Error: invalid matrix in '%s'\n",in1);return 1;}

    if(!strcmp(op,"sum")||!strcmp(op,"sub")||!strcmp(op,"mul")){
        const char*in2=argv[3];
        if(readM(in2,&B)){fprintf(stderr,"Error: invalid matrix in '%s'\n",in2);freeM(A);return 1;}
        if(!strcmp(op,"sum"))R=sumM(A,B);
        else if(!strcmp(op,"sub"))R=subM(A,B);
        else R=mulM(A,B);
        if(!R){fprintf(stderr,"Error: no solution\n");freeM(A);freeM(B);return 1;}
        writeM(out,R);
    }else if(!strcmp(op,"pow")){
        const char*pfile=argv[3];
        FILE*f=fopen(pfile,"r");
        long long p; if(!f||fscanf(f,"%lld",&p)!=1){fprintf(stderr,"Error: bad power file\n");freeM(A);return 1;}
        fclose(f);
        R=powM(A,p);
        if(!R){fprintf(stderr,"Error: no solution\n");freeM(A);return 1;}
        writeM(out,R);
    }else if(!strcmp(op,"det")){
        double d=detM(A);
        if(isnan(d)){fprintf(stderr,"Error: det failed\n");freeM(A);return 1;}
        FILE*f=fopen(out,"w");fprintf(f,"%.6f\n",d);fclose(f);
    }else{
        fprintf(stderr,"Error: unknown operation '%s'\n",op);
        freeM(A);return 1;
    }

    freeM(A);freeM(B);freeM(R);
    return 0;
}


