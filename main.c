int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: matrix <op> <input1> [input2] <output>\n");
        return 1;
    }

    const char *op = argv[1];
    const char *in1 = argv[2];
    const char *in2 = (argc == 5) ? argv[3] : NULL;
    const char *out = (argc == 5) ? argv[4] : argv[3];

    int need_two = (!strcmp(op,"sum") || !strcmp(op,"sub") || !strcmp(op,"mul"));
    int need_pow = (!strcmp(op,"pow"));
    int need_det = (!strcmp(op,"det"));

    if (!(need_two || need_pow || need_det)) {
        fprintf(stderr,"Unknown operation\n");
        return 1;
    }

    if (need_two && argc != 5) { fprintf(stderr,"Usage mismatch for %s\n",op); return 1; }
    if (need_pow && argc != 5) { fprintf(stderr,"Usage mismatch for %s\n",op); return 1; }
    if (need_det && argc != 4) { fprintf(stderr,"Usage mismatch for det\n"); return 1; }

    FILE *f1 = fopen(in1,"r");
    if (!f1) { fprintf(stderr,"Cannot open %s\n",in1); return 1; }
    Matrix *A=NULL,*B=NULL,*R=NULL;
    if (read_matrix(f1,&A)) { fprintf(stderr,"Read error in %s\n",in1); fclose(f1); return 1; }
    fclose(f1);

    if (need_two) {
        FILE *f2=fopen(in2,"r");
        if(!f2){fprintf(stderr,"Cannot open %s\n",in2); free_matrix(A); return 1;}
        if(read_matrix(f2,&B)){fprintf(stderr,"Read error in %s\n",in2); fclose(f2); free_matrix(A); return 1;}
        fclose(f2);
    }

    FILE *fo=fopen(out,"w");
    if(!fo){fprintf(stderr,"Cannot open %s\n",out); free_matrix(A); free_matrix(B); return 1;}

    int exitcode=0;

    if(!strcmp(op,"sum")){
        R=sum_matrix(A,B);
        if(!R||write_matrix(fo,R)) exitcode=1;
    } else if(!strcmp(op,"sub")){
        R=sub_matrix(A,B);
        if(!R||write_matrix(fo,R)) exitcode=1;
    } else if(!strcmp(op,"mul")){
        R=mul_matrix(A,B);
        if(!R||write_matrix(fo,R)) exitcode=1;
    } else if(!strcmp(op,"pow")){
        FILE *pf=fopen(in2,"r");
        long long p; if(!pf||fscanf(pf,"%lld",&p)!=1){exitcode=1;}
        else { fclose(pf);
            R=pow_matrix(A,p);
            if(!R||write_matrix(fo,R)) exitcode=1;
        }
    } else if(!strcmp(op,"det")){
        double d=det_matrix(A);
        if(isnan(d)) exitcode=1;
        else fprintf(fo,"%.6f\n",d);
    }

    fclose(fo);
    free_matrix(A);
    free_matrix(B);
    free_matrix(R);
    return exitcode;
}
