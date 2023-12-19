extern "C" {
__global__ void atomic_1d(const float* pos,
const long long npoints, const long long a, 
const double sampling, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints)
    return ;

    float diff_x = (pos[m] - pos[n]) * sampling;
    int offset_x = __double2int_rd((a-1)*sampling);
    int res_x = __double2int_rd(diff_x + offset_x);
    if ((res_x <= 0 || res_x >= 2*offset_x))
        return;
    atomicAdd(&output[res_x], 1.);

}

__global__ void atomic_2d(const float* pos_x, const float* pos_y,
const long long npoints, const long long a, const long long b, 
const double sampling, float* integ, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints)
    return ;

    int pos_xm = __double2int_rd(pos_x[m]);
    int pos_ym = __double2int_rd(pos_y[m]);
    int pos_xn = __double2int_rd(pos_x[n]);
    int pos_yn = __double2int_rd(pos_y[n]);


    if (n == 0) {
    	atomicAdd(&integ[pos_xm * b + pos_ym], 1.);
    	} 

    long long size = 2*b-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    float diff_x = (pos_xm - pos_xn) * sampling;
    float diff_y = (pos_ym - pos_yn) * sampling;
    int res_x = diff_x + offset_x;
    int res_y = diff_y + offset_y;
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b))
    return;
    atomicAdd(&output[res_x * size + res_y], 1.);
}

__global__ void atomic_sim(const float* pos_x, const float* pos_y,
const long long npoints, const long long a, const long long b, 
const double sampling, float* sfr, float* integ, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints)
    return ;

    int pos_xm = __double2int_rd(pos_x[m]);
    int pos_ym = __double2int_rd(pos_y[m]);
    int pos_xn = __double2int_rd(pos_x[n]);
    int pos_yn = __double2int_rd(pos_y[n]);

    float fm = sfr[pos_xm * b + pos_ym];
    float fn = sfr[pos_xn * b + pos_yn];

    if (n == 0) {
    	atomicAdd(&integ[pos_xm * b + pos_ym], 1.);
    	} 

    long long size = 2*b-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    float diff_x = (pos_xm - pos_xn) * sampling;
    float diff_y = (pos_ym - pos_yn) * sampling;
    int res_x = diff_x + offset_x;
    int res_y = diff_y + offset_y;
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b))
    return;
    atomicAdd(&output[res_x * size + res_y], fm*fn);
}

__global__ void cross_2d(const float* pos1_x, const float* pos1_y, const
        float* pos2_x, const float* pos2_y, const long long npoints, const long
        long npoints2, const long long a, const long long b, const double sampling, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints2)
    return ;

    int pos1_xm = __double2int_rd(pos1_x[m]);
    int pos1_ym = __double2int_rd(pos1_y[m]);
    int pos2_xn = __double2int_rd(pos2_x[n]);
    int pos2_yn = __double2int_rd(pos2_y[n]);


    long long size = 2*b-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    float diff_x = ((pos1_xm) - (pos2_xn)) * sampling;
    float diff_y = ((pos1_ym) - (pos2_yn)) * sampling;
    int res_x = diff_x + offset_x;
    int res_y = diff_y + offset_y;
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b))
    return;
    atomicAdd(&output[(res_x) * size + res_y], 1.);
}

__global__ void cinteg_3d(const float* pos_x, const float* pos_y, const float* pos_z, const long long npoints, const long long a, const long long b, const long long c, const double sampling, float* integ, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints)
    return ;
    
    float fm = integ[(__double2int_rd(pos_x[m]) * b +  __double2int_rd(pos_y[m]))*c + __double2int_rd(pos_z[m])];
    float fn = integ[(__double2int_rd(pos_x[n]) * b +  __double2int_rd(pos_y[n]))*c + __double2int_rd(pos_z[n])];


    long long size_y = 2*b-1 ;
    long long size_z = 2*c-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    int offset_z = __double2int_rd((c-1)*sampling);
    float diff_x = ((pos_x[m]) - (pos_x[n])) * sampling;
    float diff_y = ((pos_y[m]) - (pos_y[n])) * sampling;
    float diff_z = ((pos_z[m]) - (pos_z[n])) * sampling;
    int res_x = __double2int_rd(diff_x) + offset_x;
    int res_y = __double2int_rd(diff_y) + offset_y;
    int res_z = __double2int_rd(diff_z) + offset_z;
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b) || (res_z < 0 || res_z >= 2*c))
    return;

    atomicAdd(&output[((res_x) * size_y + res_y)*size_z + res_z], fm*fn);
}

__global__ void atomic_3d(const float* pos_x, const float* pos_y, const float* pos_z, const long long npoints, const long long c0, const long long c1, const long long a, const long long b, const long long cz, const double sampling,  float* integ, float* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints)
    return ;

    int pos_xm = __double2int_rd(pos_x[m]);
    int pos_ym = __double2int_rd(pos_y[m]);
    int pos_zm = __double2int_rd(pos_z[m]);
    int pos_xn = __double2int_rd(pos_x[n]);
    int pos_yn = __double2int_rd(pos_y[n]);
    int pos_zn = __double2int_rd(pos_z[n]);

    if (n == 0) {
    	atomicAdd(&integ[((pos_xm+c0) * b + pos_ym + c1) * cz + pos_zm], 1.);
	}
    
    long long size_y = 2*b-1 ;
    long long size_z = 2*cz-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    int offset_z = __double2int_rd((cz-1)*sampling);
    float diff_x = ((pos_xm) - (pos_xn)) * sampling;
    float diff_y = ((pos_ym) - (pos_yn)) * sampling;
    float diff_z = ((pos_zm) - (pos_zn)) * sampling;
    int res_x = __double2int_rd(diff_x) + offset_x;
    int res_y = __double2int_rd(diff_y) + offset_y;
    int res_z = __double2int_rd(diff_z) + offset_z;
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b) || (res_z < 0 || res_z >= 2*cz))
    return;

    atomicAdd(&output[((res_x) * size_y + res_y)*size_z + res_z], 1.);
}

__global__ void cross_3d(const float* pos1_x, const float* pos1_y, const float* pos1_z, const float* pos2_x, const float* pos2_y, const float* pos2_z,
        		const long long npoints, const long long npoints2, const long long a, const long long b, const long long c, const double sampling, double* output) {
    long long m = blockDim.x * blockIdx.x + threadIdx.x;
    long long n = blockDim.y * blockIdx.y + threadIdx.y;
    if (m >= npoints || n >= npoints2)
    return ;
	
    int pos1_xm = __double2int_rd(pos1_x[m]);
    int pos1_ym = __double2int_rd(pos1_y[m]);
    int pos1_zm = __double2int_rd(pos1_z[m]);
    int pos2_xn = __double2int_rd(pos2_x[n]);
    int pos2_yn = __double2int_rd(pos2_y[n]);
    int pos2_zn = __double2int_rd(pos2_z[n]);


    long long size_y = 2*b-1 ;
    long long size_z = 2*c-1 ;
    int offset_x = __double2int_rd((a-1)*sampling);
    int offset_y = __double2int_rd((b-1)*sampling);
    int offset_z = __double2int_rd((c-1)*sampling);
    
    float diff_x = ((pos1_xm) - (pos2_xn)) * sampling;
    float diff_y = ((pos1_ym) - (pos2_yn)) * sampling;
    float diff_z = ((pos1_zm) - (pos2_zn)) * sampling;
    
    int res_x = __double2int_rd(diff_x) + offset_x;
    int res_y = __double2int_rd(diff_y) + offset_y;
    int res_z = __double2int_rd(diff_z) + offset_z;
    
    if ((res_x < 0 || res_x >= 2*a) || (res_y < 0 || res_y >= 2*b) || (res_z < 0 || res_z >= 2*c))
    return;

    atomicAdd(&output[((res_x) * size_y + res_y)*size_z + res_z], 1.);
}

__global__ 
void thresh_frame(double *frame, const long long npix, const double threshold, const double *mask) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x ;
    if (pix >= npix)
        return ;
    if (frame[pix] < threshold)
        frame[pix] = 0 ;
    frame[pix] *= mask[pix] ;
}

}
