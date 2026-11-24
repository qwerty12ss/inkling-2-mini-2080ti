// Small-row tensor parallel dense projections. The regular and escape streams contain
// the same FP16 operands scaled by eight; every source FP16 bit is recoverable.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {

__device__ __forceinline__ void mma(float (&c)[4],uint32_t a0,uint32_t a1,uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                 : "r"(a0),"r"(a1),"r"(b));
}

template<int pair>
__device__ __forceinline__ uint32_t unpack(uint32_t low,uint32_t upper) {
    uint32_t lo=__byte_perm(low,0,pair%2 ? 0x4342 : 0x4140);
    uint32_t hi=(upper>>(4*pair))&0x000f000fu;
    uint32_t val=(lo|(hi<<8))<<3;
    return val+(val&0x40004000u);
}

template<int warps,bool packed,bool segments=false,class Output=half>
__global__ void project(const half* __restrict__ x,const uint4* __restrict__ raw,
                        const uint2* __restrict__ low,const uint32_t* __restrict__ upper,
                        const int* __restrict__ escape,const unsigned char* __restrict__ flags,
                        float* __restrict__ partial,Output* __restrict__ out,int m,int n,int k,
                        int splits,const half* xk=nullptr,const half* xv=nullptr) {
    __shared__ float sm[warps][8][20];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,rowgroup=lane>>2,within=lane&3;
    int groups=k/16,tile=blockIdx.x;
    int pairid=blockIdx.y*warps+warp, lanes=warps*splits;
    int begin=groups*pairid/lanes,end=groups*(pairid+1)/lanes;
    bool has_escape=packed && flags[tile]!=0;
    const int startcol=tile*16;
    const half* primary=x;
    const half* alternate=nullptr;
    int boundary=16;
    if constexpr(segments) {
        primary=startcol<3276?x:(startcol<3788?xk:xv);
        if(startcol<3276 && startcol+16>3276) {alternate=xk;boundary=3276-startcol;}
        if(startcol<3788 && startcol+16>3788) {alternate=xv;boundary=3788-startcol;}
    }
    float acc[4]={0,0,0,0},second[4]={0,0,0,0};
    #pragma unroll 2
    for(int group=begin;group<end;++group) {
        int g=tile*groups+group;
        uint2 a=rowgroup<m ? __ldg(reinterpret_cast<const uint2*>(primary+rowgroup*k+group*16+within*4)) : make_uint2(0,0);
        int special=has_escape?__ldg(escape+g):-1;
        uint4 b;
        if(!packed || special>=0) {
            int bank=packed?special:g;
            b=__ldcs(raw+bank*32+lane);
        } else {
            uint2 l=__ldcs(low+g*32+lane);
            uint32_t h=__ldcs(upper+g*32+lane);
            b=make_uint4(unpack<0>(l.x,h),unpack<1>(l.x,h),unpack<2>(l.y,h),unpack<3>(l.y,h));
        }
        mma(acc,b.x,b.y,a.x);
        mma(acc,b.z,b.w,a.y);
        if constexpr(segments) {
            if(alternate) {
                uint2 other=rowgroup<m ? __ldg(reinterpret_cast<const uint2*>(alternate+rowgroup*k+group*16+within*4)) : make_uint2(0,0);
                mma(second,b.x,b.y,other.x);
                mma(second,b.z,b.w,other.y);
            }
        }
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        int col=rowgroup+(i&2)*4;
        sm[warp][within*2+(i&1)][col]=segments && col>=boundary ? second[i] : acc[i];
    }
    __syncthreads();
    for(int idx=threadIdx.x;idx<m*16;idx+=blockDim.x) {
        int row=idx>>4,col=idx&15,off=row*n+tile*16+col;
        float result=0;
        #pragma unroll
        for(int i=0;i<warps;++i) result+=sm[i][row][col];
        if(splits==1) {
            float restored=result*(packed?.125f:1.f);
            if constexpr(std::is_same<Output,half>::value)out[off]=__float2half_rn(restored);
            else out[off]=restored;
        } else partial[blockIdx.y*m*n+off]=result;
    }
}

__global__ void reduce(const float* __restrict__ input,half* out,int count,int splits,float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    float v=0;
    for(int s=0;s<splits;++s) v+=input[s*count+i];
    out[i]=__float2half_rn(v*scale);
}
}

extern "C" int small_dense(const half* x,const uint4* raw,const uint2* low,const uint32_t* upper,
                           const int* escape,const unsigned char* flags,float* partial,half* out,
                           int m,int n,int k,int splits,int warps,int packed,cudaStream_t stream) {
    if(m<1 || m>8 || n%16 || k%16 || splits<1 || splits>16 || warps*splits>k/16) return -1;
    dim3 grid(n/16,splits);
    #define DISPATCH(W,P) project<W,P><<<grid,32*W,0,stream>>>(x,raw,low,upper,escape,flags,partial,out,m,n,k,splits)
    if(warps==4 && packed) {DISPATCH(4,true);}
    else if(warps==4) {DISPATCH(4,false);}
    else if(warps==8 && packed) {DISPATCH(8,true);}
    else if(warps==8) {DISPATCH(8,false);}
    else if(warps==16 && packed) {DISPATCH(16,true);}
    else if(warps==16) {DISPATCH(16,false);}
    else return -2;
    if(splits>1) reduce<<<(m*n+255)/256,256,0,stream>>>(partial,out,m*n,splits,packed?.125f:1.f);
    return int(cudaPeekAtLastError());
}

// This physical kernel reads each original Q/K/V weight once. Only the two
// N tiles straddling the real 3276/3788 boundaries need a second X fragment.
extern "C" int dense12_qkv(const half* x,const half* xk,const half* xv,const uint4* esc,
       const uint2* lo,const uint32_t* hi,const int* ind,const unsigned char* flag,
       float* partial,half* out,int m,cudaStream_t stream) {
    if(m<1 || m>8)return -1;
    project<8,true,true><<<4304/16,256,0,stream>>>(x,esc,lo,hi,ind,flag,partial,out,m,4304,2304,1,xk,xv);
    return int(cudaPeekAtLastError());
}

__global__ void materialize_kernel(const uint4* raw,const uint2* low,const uint32_t* upper,
              const int* escape,const unsigned char* flags,half* dst,int n,int k) {
    const int tile=blockIdx.x,groups=k/16,slot=threadIdx.x>>5,lane=threadIdx.x&31;
    const int group=blockIdx.y*8+slot;
    if(group>=groups)return;
    const int index=tile*groups+group;
    const int ex=flags[tile]?__ldg(escape+index):-1;
    uint4 words;
    if(ex>=0)words=__ldcs(raw+ex*32+lane);
    else {
        uint2 l=__ldcs(low+index*32+lane);uint32_t h=__ldcs(upper+index*32+lane);
        words=make_uint4(unpack<0>(l.x,h),unpack<1>(l.x,h),unpack<2>(l.y,h),unpack<3>(l.y,h));
    }
    uint32_t pairs[4]={words.x,words.y,words.z,words.w};
    #pragma unroll
    for(int p=0;p<4;++p) {
        int nr=tile*16+(p%2)*8+(lane>>2);
        int kc=group*16+(lane&3)*4+(p/2)*2;
        half2 restored=__float22half2_rn(make_float2(__half2float(__ushort_as_half(pairs[p]&65535))*.125f,
                                                    __half2float(__ushort_as_half(pairs[p]>>16))*.125f));
        *reinterpret_cast<half2*>(dst+nr*k+kc)=restored;
    }
}

extern "C" int dense12_materialize(const uint4* raw,const uint2* low,const uint32_t* upper,
              const int* escape,const unsigned char* flags,half* dst,int n,int k,cudaStream_t stream) {
    if(n<16 || k<16 || n%16 || k%16)return -1;
    materialize_kernel<<<dim3(n/16,(k/16+7)/8),256,0,stream>>>(raw,low,upper,escape,flags,dst,n,k);
    return int(cudaPeekAtLastError());
}
