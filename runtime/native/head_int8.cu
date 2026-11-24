// Auxiliary grouped INT8 vocabulary projection for a single 8-position Turing warp tile.
// Original FP16 weights remain available for exact candidate scoring and reference checks.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int groups=18;
constexpr int width=2304;
constexpr int group_width=128;
constexpr unsigned mask=0xffffffffu;

__global__ void pack_inputs(const half* x,int rows,signed char* q,float* scales) {
    int row=blockIdx.x/groups, group=blockIdx.x%groups, lane=threadIdx.x;
    const int offset=group*group_width+4*lane;
    float vals[4],peak=0.f;
    #pragma unroll
    for (int i=0;i<4;++i) {
        vals[i]=row<rows?__half2float(x[row*width+offset+i]):0.f;
        peak=fmaxf(peak,fabsf(vals[i]));
    }
    #pragma unroll
    for(int stride=16;stride>0;stride>>=1) peak=fmaxf(peak,__shfl_xor_sync(mask,peak,stride));
    float scale=fmaxf(peak*(1.f/127.f),1e-20f);
    if(lane==0) scales[row*groups+group]=scale;
    #pragma unroll
    for(int i=0;i<4;++i)
        q[row*width+offset+i]=static_cast<signed char>(__float2int_rn(vals[i]/scale));
}

__device__ inline void add_products(uint32_t a,uint32_t b,int& x,int& y) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                 : "+r"(x),"+r"(y) : "r"(a),"r"(b));
}

__global__ void project(const uint4* weights,const float* ws,const uint32_t* queries,
                        const float* qs,half* out,int rows,int n) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(warp>=n/8) return;
    int lane=threadIdx.x&31,row=lane>>2,within=lane&3;
    int col=warp*8+within*2;
    float sum0=0.f,sum1=0.f;
    #pragma unroll 2
    for(int g=0;g<groups;++g) {
        int accum0=0,accum1=0;
        const uint4 a=weights[((warp*groups+g)*2)*32+lane];
        const uint4 b=weights[((warp*groups+g)*2+1)*32+lane];
        const uint32_t* q=queries+row*(width/4)+g*(group_width/4)+within;
        add_products(q[0],a.x,accum0,accum1);
        add_products(q[4],a.y,accum0,accum1);
        add_products(q[8],a.z,accum0,accum1);
        add_products(q[12],a.w,accum0,accum1);
        add_products(q[16],b.x,accum0,accum1);
        add_products(q[20],b.y,accum0,accum1);
        add_products(q[24],b.z,accum0,accum1);
        add_products(q[28],b.w,accum0,accum1);
        float common=qs[row*groups+g];
        const float2 scale=reinterpret_cast<const float2*>(ws+((warp*groups+g)*8))[within];
        sum0=fmaf(static_cast<float>(accum0),common*scale.x,sum0);
        sum1=fmaf(static_cast<float>(accum1),common*scale.y,sum1);
    }
    if(row<rows) *reinterpret_cast<half2*>(out+row*n+col)=__floats2half2_rn(sum0,sum1);
}
}

extern "C" int head_projection(const void* x,int rows,const void* w,const void* scales,
                              void* q,void* qscale,void* out,int n,void* stream) {
    if(rows<1 || rows>8 || n%8!=0) return (int) cudaErrorInvalidValue;
    cudaStream_t s=reinterpret_cast<cudaStream_t>(stream);
    pack_inputs<<<8*groups,32,0,s>>>(static_cast<const half*>(x),rows,static_cast<signed char*>(q),
                                      static_cast<float*>(qscale));
    project<<<(n/8+3)/4,128,0,s>>>(static_cast<const uint4*>(w),static_cast<const float*>(scales),
        static_cast<const uint32_t*>(q),static_cast<const float*>(qscale),static_cast<half*>(out),rows,n);
    return cudaPeekAtLastError();
}
