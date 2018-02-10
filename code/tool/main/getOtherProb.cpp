#include<iostream>
#include"string.h"
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"

#define sqr(a) ((a)*(a))
#define max(a,b) ((a)>(b)?(a):(b))


double getProb(double ht,double wt,double *pos,double *prob,int cNum){
    double result,tmp,minDist=100000000;
    int c;
    if((ht<pos[0])||(wt<pos[1])||(ht>pos[cNum*2-2])||(wt>pos[cNum*2-1])){
        result=0;
        return result;
    }
    for(c=0;c<cNum;c++){
        tmp=sqr(ht-pos[c*2])+sqr(wt-pos[c*2+1]);
        if(tmp<minDist){
            minDist=tmp;
            result=prob[c];
        }
    }
    return result;
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=5)
		mexErrMsgTxt("\nErrors in input.\n");
	
    double *prob,*pos,*result,ht,wt;
    int cNum,pNum,chNum,ch,p,pt,c,ct,ch_bias,pt_bias,ct_bias,p_bias;
    
    //readinfo
    prob=((double*)mxGetPr(prhs[0]));
    pos=((double*)mxGetPr(prhs[1]));
    cNum=int(*((double*)mxGetPr(prhs[2])));
    pNum=int(*((double*)mxGetPr(prhs[3])));
    chNum=int(*((double*)mxGetPr(prhs[4])));
    
    if(mxGetN(prhs[1])!=cNum*pNum*chNum)
        mexErrMsgTxt("\nErrors in input.\n");
    
    //output memory
	plhs[0]=mxCreateDoubleMatrix(pNum,cNum*pNum*chNum,mxREAL);
    result=mxGetPr(plhs[0]);

    //processing
    memset(result,0,sizeof(double)*pNum*cNum*pNum*chNum);
    for(ch=0;ch<chNum;ch++){
        ch_bias=cNum*pNum*ch;
        for(pt=0;pt<pNum;pt++){
            pt_bias=cNum*pt+ch_bias;
            for(ct=0;ct<cNum;ct++){
                ct_bias=ct+pt_bias;
                ht=pos[ct_bias*2];
                wt=pos[ct_bias*2+1];
                for(p=0;p<pNum;p++){
                    p_bias=cNum*p+ch_bias;
                    result[p+ct_bias*pNum]=getProb(ht,wt,pos+2*p_bias,prob+p_bias,cNum);
                }
            }
        }
    }
}
