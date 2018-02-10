#include<iostream>
#include"string.h"
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"

#define sqr(a) ((a)*(a))
#define max(a,b) ((a)>(b)?(a):(b))
#define pi 3.1415926536
#define tooFar 3.0

double sqrt2pi=sqrt(2*pi);

inline double getGauss(double dist,double delta){
    return exp(-sqr(dist)/(2*sqr(delta)))/(sqrt2pi*delta);
}


inline double getGauss_sqr(double sqrdist,double var1,double var2){
    return exp(-sqrdist/var1)/var2;
}


inline double getLogGaussDer(double diff,double delta){
    return -diff/sqr(delta);
}


void getGaussTemp(double** GaussTemp,double halfRange,double* posCand,int posNum,double delta){
    int ih,iw,jh,jw,ic,jc;
    double pH,pW,sqrdist_tmp,sqrdist,var1,var2,tooFarSqrDist,tooSmall;
    var1=2*sqr(delta);
    var2=sqrt2pi*delta;
    tooFarSqrDist=sqr(delta*tooFar);
    tooSmall=exp(-sqr(tooFar)/2.0)/var2;
    ic=0;
    for(iw=0;iw<posNum;iw++){
        pW=posCand[iw];
        for(ih=0;ih<posNum;ih++){
            pH=posCand[ih];
            jc=0;
            memset(GaussTemp[ic],0,sizeof(double)*sqr(halfRange*2+1));
            for(jw=-halfRange;jw<=halfRange;jw++){
                sqrdist_tmp=sqr(pW-jw);
                for(jh=-halfRange;jh<=halfRange;jh++){
                    sqrdist=sqrdist_tmp+sqr(pH-jh);
                    if(sqrdist<tooFarSqrDist)
                        GaussTemp[ic][jc]=getGauss_sqr(sqrdist,var1,var2);
                    else
                        GaussTemp[ic][jc]=tooSmall;
                    jc++;
                }
            }
            ic++;
        }
    }
}


void getResponseMap(double *map,double *x,double h,double w,double halfRange,int len){
    int c,i,j;
    h=round(h);
    w=round(w);
    c=0;
    for(i=w-halfRange;i<=w+halfRange;i++){
        for(j=h-halfRange;j<=h+halfRange;j++){
            if((i<0)||(i>=len)||(j<0)||(j>=len))
                map[c]=0;
            else
                map[c]=max(x[i*len+j],0);
            c++;
        }
    }
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=7)
		mexErrMsgTxt("\nErrors in input.\n");
	
    double *x,*coord,*prob,*prior,*derLog,*pos,*posCand,*oriPosCand,*halfRange,**GaussTemp,*map,map_delta,*deform_delta,sum,deform1,deform2,der_h,der_w,pos_h,pos_w,h,w;
    int posNum,patNum,len,pixelNum,posTotal,channelNum,coord_bias,x_bias,prob_bias,derLog_bias,pos_bias,maxPixelNum;
    int ch,i,j,k,p1,p2,idx;
    
    //readinfo
    x=((double*)mxGetPr(prhs[0]));
    coord=((double*)mxGetPr(prhs[1]));
    oriPosCand=((double*)mxGetPr(prhs[2]));
    halfRange=((double*)mxGetPr(prhs[3]));
    deform_delta=((double*)mxGetPr(prhs[4]));
    map_delta=*((double*)mxGetPr(prhs[5]));
    channelNum=int(*((double*)mxGetPr(prhs[6])));
    len=mxGetM(prhs[0]);
    patNum=mxGetN(prhs[1])/channelNum;
    if((mxGetN(prhs[1])%channelNum!=0)||(mxGetN(prhs[0])!=len*channelNum))
        mexErrMsgTxt("\nErrors in input.\n");
    posNum=mxGetM(prhs[2]);
    posTotal=sqr(posNum);
    
    //output memory
	plhs[0]=mxCreateDoubleMatrix(posTotal,patNum*channelNum,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(posTotal,patNum*channelNum,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(2*posTotal,patNum*channelNum,mxREAL);
    plhs[3]=mxCreateDoubleMatrix(2*posTotal,patNum*channelNum,mxREAL);
    
    prob=mxGetPr(plhs[0]);
    prior=mxGetPr(plhs[1]);
    derLog=mxGetPr(plhs[2]);
    pos=mxGetPr(plhs[3]);
    while(GaussTemp=new double*[posTotal],GaussTemp==NULL);
    maxPixelNum=0;
    for(i=0;i<patNum;i++)
        maxPixelNum=max(maxPixelNum,sqr(halfRange[i]*2+1));
    for(i=0;i<sqr(posNum);i++)
        while(GaussTemp[i]=new double[maxPixelNum],GaussTemp[i]==NULL);
    while(map=new double[maxPixelNum],map==NULL);
    
    //processing
    memset(prob,0,sizeof(double)*posTotal*patNum*channelNum);
    memset(prior,0,sizeof(double)*posTotal*patNum*channelNum);
    
    for(ch=0;ch<channelNum;ch++){
        coord_bias=ch*2*patNum;
        x_bias=ch*sqr(len);
        prob_bias=ch*posTotal*patNum;
        derLog_bias=ch*2*posTotal*patNum;
        pos_bias=ch*2*posTotal*patNum;
        for(i=0;i<patNum;i++){
            posCand=oriPosCand+(ch*patNum+i)*posNum;
            pixelNum=sqr(halfRange[i]*2+1);
            getGaussTemp(GaussTemp,halfRange[i],posCand,posNum,map_delta);
            h=coord[i*2+coord_bias]-1;
            w=coord[i*2+1+coord_bias]-1;
            getResponseMap(map,x+x_bias,h,w,halfRange[i],len);
            j=0;
            for(p1=0;p1<posNum;p1++){
                deform1=getGauss(posCand[p1],deform_delta[i]);
                der_w=getLogGaussDer(-posCand[p1],deform_delta[i]);
                pos_w=w+1+posCand[p1];
                for(p2=0;p2<posNum;p2++){
                    deform2=getGauss(posCand[p2],deform_delta[i]);
                    der_h=getLogGaussDer(-posCand[p2],deform_delta[i]);
                    pos_h=h+1+posCand[p2];
                    sum=0;
                    for(k=0;k<pixelNum;k++)
                        sum+=GaussTemp[j][k]*map[k];
                    prob[i*posTotal+j+prob_bias]=deform1*deform2;
                    prior[i*posTotal+j+prob_bias]=sum;
                    idx=i*2*posTotal+j*2+derLog_bias;
                    derLog[idx]=der_h;
                    derLog[idx+1]=der_w;
                    idx=i*2*posTotal+j*2+pos_bias;
                    pos[idx]=pos_h;
                    pos[idx+1]=pos_w;
                    j++;
                }
            }
            
            
        }

        /*double *tmp;
        plhs[1]=mxCreateDoubleMatrix(sqr(halfRange*2+1),sqr(posNum),mxREAL);
        tmp=mxGetPr(plhs[1]);
        int j,c=0;
        for(i=0;i<sqr(posNum);i++)
            for(j=0;j<sqr(halfRange*2+1);j++,c++)
                tmp[c]=GaussTemp[i][j];*/


        /*for(i=0;i<dimNum;i++){
            theP=prob[i];
            for(o1=0;o1<objNum;o1++){
                idx_1=i+o1*dimNum;
                //mexPrintf("%d %d\n",idx,idx_1);
                norm[o1]+=sqr(fea[idx_1])*theP;
                norm_flip[o1]+=sqr(fea_flip[idx_1])*theP;
                for(o2=0;o2<objNum;o2++){
                    idx_2=i+o2*dimNum;
                    idx=o1+o2*objNum;
                    //mexPrintf("%d %d %d\n",idx,idx_1,idx_2);
                    cosdist[idx]+=fea[idx_2]*fea[idx_1]*theP;
                    cosdist_flip[idx]+=fea[idx_2]*fea_flip[idx_1]*theP;
                }
            }
        }*/
    }
    for(i=0;i<sqr(posNum);i++)
        delete GaussTemp[i];
    delete[] GaussTemp;
    delete[] map;
}
