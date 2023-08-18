
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include <sstream>
#include <cmath>

namespace libpressio { namespace ssim_metrics_ns {

 /* Code taken from CODARCode/qcat
 *  (C) 2015 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#define K1 0.01
#define K2 0.03
#define QCAT_FLOAT 0
#define QCAT_DOUBLE 1
#define QCAT_INT32 2
#define QCAT_INT16 3
#define QCAT_UINT32 4
#define QCAT_UINT16 5
    namespace ssim {
void throw_size_error(size_t a, size_t b) {
    std::stringstream ss;
    ss << "ERROR windowSize = " << a << " > " << b;
    throw std::runtime_error(ss.str());

}
double SSIM_1d_calcWindow_float(float* data, float* other, size_t offset0, size_t windowSize0);
double SSIM_1d_calcWindow_double(double* data, double* other, size_t offset0, size_t windowSize0);
double SSIM_1d_windowed_float(float* oriData, float* decData, size_t size0, size_t windowSize0, size_t windowShift0);
double SSIM_1d_windowed_double(double* oriData, double* decData, size_t size0, size_t windowSize0, size_t windowShift0);

double SSIM_2d_calcWindow_float(float* data, float *other, size_t size0, size_t offset0, size_t offset1, size_t windowSize0, size_t windowSize1);
double SSIM_2d_calcWindow_double(double* data, double *other, size_t size0, size_t offset0, size_t offset1, size_t windowSize0, size_t windowSize1);
double SSIM_2d_windowed_float(float* oriData, float* decData, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowShift0, size_t windowShift1);double SSIM_2d_windowed_double(double* oriData, double* decData, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowShift0, size_t windowShift1);

double SSIM_3d_calcWindow_float(float* data, float* other, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t windowSize0, size_t windowSize1, size_t windowSize2);
double SSIM_3d_calcWindow_double(double* data, double* other, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t windowSize0, size_t windowSize1, size_t windowSize2);
double SSIM_3d_windowed_float(float* oriData, float* decData, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowShift0, size_t windowShift1, size_t windowShift2);
double SSIM_3d_windowed_double(double* oriData, double* decData, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowShift0, size_t windowShift1, size_t windowShift2);

double SSIM_4d_calcWindow_float(float* data, float* other, size_t size2, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t offset3,size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3);
double SSIM_4d_calcWindow_double(double* data, double* other, size_t size2, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t offset3,size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3);
double SSIM_4d_windowed_float(float* oriData, float* decData, size_t size3, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3, size_t windowShift0, size_t windowShift1, size_t windowShift2, size_t windowShift3);
double SSIM_4d_windowed_double(double* oriData, double* decData, size_t size3, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3, size_t windowShift0, size_t windowShift1, size_t windowShift2, size_t windowShift3);

double SSIM_1d_calcWindow_float(float* data, float* other, size_t offset0, size_t windowSize0)
{
    size_t i0;
    size_t np=0; //Number of points

    float xMin=data[offset0];
    float xMax=data[offset0];
    float yMin=other[offset0];
    float yMax=other[offset0];
    double xSum=0;
    double ySum=0;

    for(i0=offset0; i0<offset0+windowSize0; i0++) {
        np++;
        if(xMin>data[i0])
            xMin=data[i0];
        if(xMax<data[i0])
            xMax=data[i0];
        if(yMin>other[i0])
            yMin=other[i0];
        if(yMax<other[i0])
            yMax=other[i0];
        xSum+=data[i0];
        ySum+=other[i0];
    }


    double xMean=xSum/(double)np;
    double yMean=ySum/(double)np;

    double var_x = 0, var_y = 0, var_xy = 0;

    for(i0=offset0; i0<offset0+windowSize0; i0++) {
        var_x += (data[i0] - xMean)*(data[i0] - xMean);
        var_y += (other[i0] - yMean)*(other[i0] - yMean);
        var_xy += (data[i0] - xMean)*(other[i0] - yMean);
    }

    var_x /= (double)np;
    var_y /= (double)np;
    var_xy /= (double)np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;

    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

double SSIM_1d_calcWindow_double(double* oriData, double* decData, size_t offset0, size_t windowSize0)
{
    size_t i0;
    size_t np=0; //Number of points

    double* data = oriData;
    double* other = decData;

    double xMin=data[offset0];
    double xMax=data[offset0];
    double yMin=other[offset0];
    double yMax=other[offset0];
    double xSum=0;
    double ySum=0;

    for(i0=offset0; i0<offset0+windowSize0; i0++) {
        np++;
        if(xMin>data[i0])
            xMin=data[i0];
        if(xMax<data[i0])
            xMax=data[i0];
        if(yMin>other[i0])
            yMin=other[i0];
        if(yMax<other[i0])
            yMax=other[i0];
        xSum+=data[i0];
        ySum+=other[i0];
    }


    double xMean=xSum/(double)np;
    double yMean=ySum/(double)np;

    double var_x = 0, var_y = 0, var_xy = 0;

    for(i0=offset0; i0<offset0+windowSize0; i0++) {
        var_x += (data[i0] - xMean)*(data[i0] - xMean);
        var_y += (other[i0] - yMean)*(other[i0] - yMean);
        var_xy += (data[i0] - xMean)*(other[i0] - yMean);
    }

    var_x /= (double)np;
    var_y /= (double)np;
    var_xy /= (double)np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;

    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

double SSIM_1d_windowed_float(float* oriData, float* decData, size_t size0, size_t windowSize0, size_t windowShift0) {
    size_t offset0;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }

    //offsetInc0=windowSize0/2;
    offsetInc0=windowShift0;


    for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
        nw++;
        ssimSum+=SSIM_1d_calcWindow_float(oriData, decData, offset0, windowSize0);
    }

    return ssimSum/nw;
}

double SSIM_1d_windowed_double(double* oriData, double* decData, size_t size0, size_t windowSize0, size_t windowShift0)
{
    size_t offset0;
    int nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }

    //offsetInc0=windowSize0/2;
    offsetInc0=windowShift0;


    for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
        nw++;
        ssimSum+=SSIM_1d_calcWindow_double(oriData, decData, offset0, windowSize0);
    }

    return ssimSum/nw;
}

//////////////////// 2D

double SSIM_2d_windowed_float(float* oriData, float* decData, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowShift0, size_t windowShift1)
{
    size_t offset0,offset1;
    int nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1;

    float* data = oriData;
    float* other = decData;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;

    for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

        for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
            nw++;
            double ssim = SSIM_2d_calcWindow_float(data, other, size0, offset0, offset1, windowSize0, windowSize1);
            ssimSum+=ssim;
        }
    }

    return ssimSum/nw;
}

double SSIM_2d_calcWindow_float(float* data, float *other, size_t size0, size_t offset0, size_t offset1, size_t windowSize0, size_t windowSize1)
{
    size_t i0,i1,index;
    size_t np=0; //Number of points
    float xMin=data[offset0+size0*offset1];
    float xMax=data[offset0+size0*offset1];
    float yMin=other[offset0+size0*offset1];
    float yMax=other[offset0+size0*offset1];
    double xSum=0;
    double ySum=0;

    for(i1=offset1; i1<offset1+windowSize1; i1++) {
        for(i0=offset0; i0<offset0+windowSize0; i0++) {
            np++;
            index=i0+size0*i1;
            if(xMin>data[index])
                xMin=data[index];
            if(xMax<data[index])
                xMax=data[index];
            if(yMin>other[index])
                yMin=other[index];
            if(yMax<other[index])
                yMax=other[index];
            xSum+=data[index];
            ySum+=other[index];
        }
    }

    double xMean=xSum/np;
    double yMean=ySum/np;

    double var_x = 0, var_y = 0, var_xy = 0;

    for(i1=offset1; i1<offset1+windowSize1; i1++) {
        for(i0=offset0; i0<offset0+windowSize0; i0++) {
            index=i0+size0*i1;
            var_x += (data[index] - xMean)*(data[index] - xMean);
            var_y += (other[index] - yMean)*(other[index] - yMean);
            var_xy += (data[index] - xMean)*(other[index] - yMean);
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;


    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

double SSIM_2d_windowed_double(double* oriData, double* decData, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowShift0, size_t windowShift1)
{
    size_t offset0,offset1;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1;

    double* data = oriData;
    double* other = decData;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;

    for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

        for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
            nw++;
            ssimSum+=SSIM_2d_calcWindow_double(data, other, size0, offset0, offset1, windowSize0, windowSize1);
        }
    }

    return ssimSum/nw;
}

double SSIM_2d_calcWindow_double(double* data, double *other, size_t size0, size_t offset0, size_t offset1, size_t windowSize0, size_t windowSize1) {
    size_t i0,i1,index;
    size_t np=0; //Number of points
    double xMin=data[offset0+size0*offset1];
    double xMax=data[offset0+size0*offset1];
    double yMin=other[offset0+size0*offset1];
    double yMax=other[offset0+size0*offset1];
    double xSum=0;
    double ySum=0;

    for(i1=offset1; i1<offset1+windowSize1; i1++) {
        for(i0=offset0; i0<offset0+windowSize0; i0++) {
            np++;
            index=i0+size0*i1;
            if(xMin>data[index])
                xMin=data[index];
            if(xMax<data[index])
                xMax=data[index];
            if(yMin>other[index])
                yMin=other[index];
            if(yMax<other[index])
                yMax=other[index];
            xSum+=data[index];
            ySum+=other[index];
        }
    }

    double xMean=xSum/np;
    double yMean=ySum/np;

    double var_x = 0, var_y = 0, var_xy = 0;

    for(i1=offset1; i1<offset1+windowSize1; i1++) {
        for(i0=offset0; i0<offset0+windowSize0; i0++) {
            index=i0+size0*i1;
            var_x += (data[index] - xMean)*(data[index] - xMean);
            var_y += (other[index] - yMean)*(other[index] - yMean);
            var_xy += (data[index] - xMean)*(other[index] - yMean);
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;

    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    
    if(ssim<0)
		printf("ssim=%f", ssim);

    return ssim;
}


//////////////////////// 3D

double SSIM_3d_windowed_float(float* oriData, float* decData, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowShift0, size_t windowShift1, size_t windowShift2)
{
    size_t offset0,offset1,offset2;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1,offsetInc2;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }
    if(windowSize2>size2) {
         throw_size_error(windowSize2, size2);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    //offsetInc2=windowSize2/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;
    offsetInc2=windowShift2;

    for(offset2=0; offset2+windowSize2<=size2; offset2+=offsetInc2) { //MOVING WINDOW

        for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

            for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
                nw++;
                ssimSum+=SSIM_3d_calcWindow_float(oriData, decData, size1, size0, offset0, offset1, offset2, windowSize0, windowSize1, windowSize2);

            }
        }
    }

    return ssimSum/nw;
}

double SSIM_3d_calcWindow_float(float* data, float* other, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t windowSize0, size_t windowSize1, size_t windowSize2) {
    size_t i0,i1,i2,index;
    size_t np=0; //Number of points
    float xMin=data[offset0+size0*(offset1+size1*offset2)];
    float xMax=data[offset0+size0*(offset1+size1*offset2)];
    float yMin=other[offset0+size0*(offset1+size1*offset2)];
    float yMax=other[offset0+size0*(offset1+size1*offset2)];
    double xSum=0;
    double ySum=0;

    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                np++;
                index=i0+size0*(i1+size1*i2);
                if(xMin>data[index])
                    xMin=data[index];
                if(xMax<data[index])
                    xMax=data[index];
                if(yMin>other[index])
                    yMin=other[index];
                if(yMax<other[index])
                    yMax=other[index];
                xSum+=data[index];
                ySum+=other[index];
            }
        }
    }


    double xMean=xSum/np;
    double yMean=ySum/np;
    double var_x = 0, var_y = 0, var_xy = 0;

    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                index=i0+size0*(i1+size1*i2);
                var_x += (data[index] - xMean)*(data[index] - xMean);
                var_y += (other[index] - yMean)*(other[index] - yMean);
                var_xy += (data[index] - xMean)*(other[index] - yMean);
            }
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;


    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;

    return ssim;
}


double SSIM_3d_windowed_double(double* oriData, double* decData, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowShift0, size_t windowShift1, size_t windowShift2)
{
    size_t offset0,offset1,offset2;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1,offsetInc2;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }
    if(windowSize2>size2) {
         throw_size_error(windowSize2, size2);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    //offsetInc2=windowSize2/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;
    offsetInc2=windowShift2;

    for(offset2=0; offset2+windowSize2<=size2; offset2+=offsetInc2) { //MOVING WINDOW

        for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

            for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
                nw++;
                ssimSum+=SSIM_3d_calcWindow_double(oriData, decData, size1, size0, offset0, offset1, offset2, windowSize0, windowSize1, windowSize2);

            }
        }
    }

    return ssimSum/nw;
}

double SSIM_3d_calcWindow_double(double* data, double* other, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t windowSize0, size_t windowSize1, size_t windowSize2)
{
    size_t i0,i1,i2,index;
    size_t np=0; //Number of points
    double xMin=data[offset0+size0*(offset1+size1*offset2)];
    double xMax=data[offset0+size0*(offset1+size1*offset2)];
    double yMin=other[offset0+size0*(offset1+size1*offset2)];
    double yMax=other[offset0+size0*(offset1+size1*offset2)];
    double xSum=0;
    double ySum=0;

    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                np++;
                index=i0+size0*(i1+size1*i2);
                if(xMin>data[index])
                    xMin=data[index];
                if(xMax<data[index])
                    xMax=data[index];
                if(yMin>other[index])
                    yMin=other[index];
                if(yMax<other[index])
                    yMax=other[index];
                xSum+=data[index];
                ySum+=other[index];
            }
        }
    }

    double xMean=xSum/np;
    double yMean=ySum/np;
    double var_x = 0, var_y = 0, var_xy = 0;

    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                index=i0+size0*(i1+size1*i2);
                var_x += (data[index] - xMean)*(data[index] - xMean);
                var_y += (other[index] - yMean)*(other[index] - yMean);
                var_xy += (data[index] - xMean)*(other[index] - yMean);
            }
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;


    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;

    return ssim;
}

double SSIM_4d_windowed_float(float* oriData, float* decData, size_t size3, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3, size_t windowShift0, size_t windowShift1, size_t windowShift2, size_t windowShift3)
{
    size_t offset0,offset1,offset2,offset3;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1,offsetInc2,offsetInc3;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }
    if(windowSize2>size2) {
         throw_size_error(windowSize2, size2);
    }
    if(windowSize3>size3) {
         throw_size_error(windowSize3, size3);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    //offsetInc2=windowSize2/2;
    //offsetInc3=windowSize3/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;
    offsetInc2=windowShift2;
    offsetInc3=windowShift3;

    for(offset3=0; offset3+windowSize3<=size3; offset3+=offsetInc3) { //MOVING WINDOW

        for(offset2=0; offset2+windowSize2<=size2; offset2+=offsetInc2) { //MOVING WINDOW

            for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

                for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
                    nw++;
                    ssimSum+=SSIM_4d_calcWindow_float(oriData, decData, size2, size1, size0, offset0, offset1, offset2, offset3, windowSize0, windowSize1, windowSize2, windowSize3);
                }
            }
        }
    }

    return ssimSum/nw;
    return 0;
}

double SSIM_4d_calcWindow_float(float* data, float* other, size_t size2, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t offset3,size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3)
{
    size_t i0,i1,i2,i3,index;
    size_t np=0; //Number of points
    float xMin=data[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    float xMax=data[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    float yMin=other[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    float yMax=other[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    double xSum=0;
    double ySum=0;
    for(i3=offset3; i3<offset3+windowSize3; i3++) {
        for(i2=offset2; i2<offset2+windowSize2; i2++) {
            for(i1=offset1; i1<offset1+windowSize1; i1++) {
                for(i0=offset0; i0<offset0+windowSize0; i0++) {
                    np++;
                    index=i0+size0*(i1+size1*(i2+size2*i3));
                    if(xMin>data[index])
                        xMin=data[index];
                    if(xMax<data[index])
                        xMax=data[index];
                    if(yMin>other[index])
                        yMin=other[index];
                    if(yMax<other[index])
                        yMax=other[index];
                    xSum+=data[index];
                    ySum+=other[index];
                }
            }
        }
    }

    double xMean=xSum/np;
    double yMean=ySum/np;
    double var_x = 0, var_y = 0, var_xy = 0;

    for(i3=offset3; i3<offset3+windowSize3; i3++) {
        for(i2=offset2; i2<offset2+windowSize2; i2++) {
            for(i1=offset1; i1<offset1+windowSize1; i1++) {
                for(i0=offset0; i0<offset0+windowSize0; i0++) {
                    index=i0+size0*(i1+size1*(i2+size2*i3));
                    var_x += (data[index] - xMean)*(data[index] - xMean);
                    var_y += (other[index] - yMean)*(other[index] - yMean);
                    var_xy += (data[index] - xMean)*(other[index] - yMean);
                }
            }
        }
    }

    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;

    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

double SSIM_4d_windowed_double(double* oriData, double* decData, size_t size3, size_t size2, size_t size1, size_t size0, size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3, size_t windowShift0, size_t windowShift1, size_t windowShift2, size_t windowShift3)
{
    size_t offset0,offset1,offset2,offset3;
    size_t nw=0; //Number of windows
    double ssimSum=0;
    size_t offsetInc0,offsetInc1,offsetInc2,offsetInc3;

    if(windowSize0>size0) {
         throw_size_error(windowSize0, size0);
    }
    if(windowSize1>size1) {
         throw_size_error(windowSize1, size1);
    }
    if(windowSize2>size2) {
         throw_size_error(windowSize2, size2);
    }
    if(windowSize3>size3) {
         throw_size_error(windowSize3, size3);
    }

    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    //offsetInc2=windowSize2/2;
    //offsetInc3=windowSize3/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;
    offsetInc2=windowShift2;
    offsetInc3=windowShift3;

    for(offset3=0; offset3+windowSize3<=size3; offset3+=offsetInc3) { //MOVING WINDOW

        for(offset2=0; offset2+windowSize2<=size2; offset2+=offsetInc2) { //MOVING WINDOW

            for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW

                for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
                    nw++;
                    ssimSum+=SSIM_4d_calcWindow_double(oriData, decData, size2, size1, size0, offset0, offset1, offset2, offset3, windowSize0, windowSize1, windowSize2, windowSize3);
                }
            }
        }
    }

    return ssimSum/nw;
    return 0;
}

double SSIM_4d_calcWindow_double(double* data, double* other, size_t size2, size_t size1, size_t size0, size_t offset0, size_t offset1, size_t offset2, size_t offset3,size_t windowSize0, size_t windowSize1, size_t windowSize2, size_t windowSize3)
{
    size_t i0,i1,i2,i3,index;
    size_t np=0; //Number of points
    double xMin=data[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    double xMax=data[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    double yMin=other[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    double yMax=other[offset0+size0*(offset1+size1*(offset2+size2*offset3))];
    double xSum=0;
    double ySum=0;
    for(i3=offset3; i3<offset3+windowSize3; i3++) {
        for(i2=offset2; i2<offset2+windowSize2; i2++) {
            for(i1=offset1; i1<offset1+windowSize1; i1++) {
                for(i0=offset0; i0<offset0+windowSize0; i0++) {
                    np++;
                    index=i0+size0*(i1+size1*(i2+size2*i3));
                    if(xMin>data[index])
                        xMin=data[index];
                    if(xMax<data[index])
                        xMax=data[index];
                    if(yMin>other[index])
                        yMin=other[index];
                    if(yMax<other[index])
                        yMax=other[index];
                    xSum+=data[index];
                    ySum+=other[index];
                }
            }
        }
    }

    double xMean=xSum/np;
    double yMean=ySum/np;
    double var_x = 0, var_y = 0, var_xy = 0;

    for(i3=offset3; i3<offset3+windowSize3; i3++) {
        for(i2=offset2; i2<offset2+windowSize2; i2++) {
            for(i1=offset1; i1<offset1+windowSize1; i1++) {
                for(i0=offset0; i0<offset0+windowSize0; i0++) {
                    index=i0+size0*(i1+size1*(i2+size2*i3));
                    var_x += (data[index] - xMean)*(data[index] - xMean);
                    var_y += (other[index] - yMean)*(other[index] - yMean);
                    var_xy += (data[index] - xMean)*(other[index] - yMean);
                }
            }
        }
    }
    var_x /= np;
    var_y /= np;
    var_xy /= np;

    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;

    double c1,c2;
    if(xMax-xMin==0) {
        c1=K1*K1;
        c2=K2*K2;
    } else {
        c1=K1*K1*(xMax-xMin)*(xMax-xMin);
        c2=K2*K2*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;

    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

int computeDimension(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	if(r1==0)
		return 0;
	else if(r2 == 0)
		return 1;
	else if(r3 == 0)
		return 2;
	else if(r4 == 0)
		return 3;
	else if(r5 == 0)
		return 4;
	return 5;
}

double calculateSSIM(void* oriData, void* decData, int dataType, size_t r4, size_t r3, size_t r2, size_t r1)
{
	int dim = computeDimension(0, r4, r3, r2, r1);
	
	size_t windowSize0 = 7;
	size_t windowSize1 = 7;
	size_t windowSize2 = 7;
	size_t windowSize3 = 7;
	
	size_t windowShift0 = 2;
	size_t windowShift1 = 2;
	size_t windowShift2 = 2;
	size_t windowShift3 = 2;
	
	double result = -1;
	if(dataType==QCAT_FLOAT) //float type
	{
		switch(dim)
		{
		case 1:
			result = SSIM_1d_windowed_float((float*)oriData, (float*)decData, r1, windowSize0, windowShift0);
			break;
		case 2:
			result = SSIM_2d_windowed_float((float*)oriData, (float*)decData, r2, r1, windowSize0, windowSize1, windowShift0, windowShift1);
			break;
		case 3:
			result = SSIM_3d_windowed_float((float*)oriData, (float*)decData, r3, r2, r1, windowSize0, windowSize1, windowSize2, windowShift0, windowShift1, windowShift2);
			break;
		case 4:
			result = SSIM_4d_windowed_float((float*)oriData, (float*)decData, r4, r3, r2, r1, windowSize0, windowSize1, windowSize2, windowSize3, windowShift0, windowShift1, windowShift2, windowShift3);
			break;
		}
	}
	else //double type
	{
		switch(dim)
		{
		case 1:
			result = SSIM_1d_windowed_double((double*)oriData, (double*)decData, r1, windowSize0, windowShift0);
			break;
		case 2:
			result = SSIM_2d_windowed_double((double*)oriData, (double*)decData, r2, r1, windowSize0, windowSize1, windowShift0, windowShift1);
			break;
		case 3:
			result = SSIM_3d_windowed_double((double*)oriData, (double*)decData, r3, r2, r1, windowSize0, windowSize1, windowSize2, windowShift0, windowShift1, windowShift2);
			break;
		case 4:
			result = SSIM_4d_windowed_double((double*)oriData, (double*)decData, r4, r3, r2, r1, windowSize0, windowSize1, windowSize2, windowSize3, windowShift0, windowShift1, windowShift2, windowShift3);
			break;
		}		
	}
	return result;
}





}


class ssim_plugin : public libpressio_metrics_plugin {
  public:
    int begin_compress_impl(struct pressio_data const* input, pressio_data const*) override {
      input_data = pressio_data::clone(*input);
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int rc) override {

      if(rc > 0 || output == nullptr) return 0;
      int datatype = 0;
      if(output->dtype() == pressio_float_dtype) datatype = QCAT_FLOAT;
      else if(output->dtype() == pressio_double_dtype) datatype = QCAT_DOUBLE;
      else return 0;

      auto norm_dims = output->normalized_dims(4);
      result = ssim::calculateSSIM(input_data.data(), output->data(), datatype, norm_dims[3], norm_dims[2], norm_dims[1], norm_dims[0]);

      return 0;
    }


  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"(computes the SSIM as implmemented in in QCAT

    SSIM the structual similar image metric
    
    https://github.com/szcompressor/qcat
    )");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "ssim:ssim", result);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<ssim_plugin>(*this);
  }
  const char* prefix() const override {
    return "ssim";
  }

  private:

  compat::optional<double> result;
  pressio_data input_data;
};

static pressio_register metrics_ssim_plugin(metrics_plugins(), "ssim", [](){ return compat::make_unique<ssim_plugin>(); });
}}

