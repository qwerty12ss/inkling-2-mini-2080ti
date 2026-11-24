// Deterministic Marlin routing on SM75. Each route's rank is its lexical
// (tile, warp, lane) position among routes to the same expert.
#include <cuda_runtime.h>

namespace {
constexpr int experts = 128;
constexpr unsigned full = 0xffffffffu;

__device__ int warp_scan(int x) {
    for (int d=1; d<32; d<<=1) {
        int prev=__shfl_up_sync(full,x,d);
        if ((threadIdx.x&31)>=d) x+=prev;
    }
    return x;
}

template<int block>
__device__ void assign_experts(int count,int *starts,int *warp_sum,int *owners,int *length) {
    // Call from the entire CTA. Four warps own the 128 experts.
    int t=threadIdx.x;
    int inclusive=0,tiles=(count+block-1)/block;
    if (t<experts) {
        inclusive=warp_scan(tiles);
        if ((t&31)==31) warp_sum[t/32]=inclusive;
    }
    __syncthreads();
    if (t<experts) {
        int base=inclusive-tiles;
        for (int w=0; w<t/32; ++w) base+=warp_sum[w];
        starts[t]=base*block;
        for (int i=0; i<tiles; ++i) owners[base+i]=t;
        if (t==experts-1) *length=(base+tiles)*block;
    }
    __syncthreads();
}

template<int threads,int block>
__global__ void small(const int *ids,int slots,int *out,int bound,int *owners,int *length) {
    constexpr int warps=threads/32;
    __shared__ int hist[warps][experts];
    __shared__ int starts[experts], sums[4];
    int t=threadIdx.x,w=t/32,lane=t&31;
    for (int j=t; j<warps*experts; j+=threads) hist[j/experts][j%experts]=0;
    for (int j=t; j<bound; j+=threads) out[j]=slots;
    for (int j=t; j<(bound+block-1)/block; j+=threads) owners[j]=-1;
    __syncthreads();
    int e=t<slots ? ids[t] : -1;
    unsigned mask=__match_any_sync(full,e);
    if (unsigned(e)<experts && lane==__ffs(mask)-1) hist[w][e]=__popc(mask);
    __syncthreads();
    int total=0;
    if (t<experts) {
        for (int n=0; n<warps; ++n) {
            int a=hist[n][t];
            hist[n][t]=total;
            total+=a;
        }
    }
    assign_experts<block>(total,starts,sums,owners,length);
    if (unsigned(e)<experts) {
        unsigned earlier=(1u<<lane)-1;
        out[starts[e]+hist[w][e]+__popc(mask&earlier)]=t;
    }
}

template<int block>
__global__ void histogram(const int *ids,int slots,int tiles,int *grid,int *out,int bound,int *owners) {
    __shared__ int counts[experts];
    int t=threadIdx.x,id=blockIdx.x;
    if (id>=tiles) {
        int p=(id-tiles)*blockDim.x+t;
        if (p<bound) out[p]=slots;
        if (p<(bound+block-1)/block) owners[p]=-1;
        return;
    }
    if (t<experts) counts[t]=0;
    __syncthreads();
    int pos=id*blockDim.x+t;
    int e=pos<slots?ids[pos]:-1;
    unsigned mask=__match_any_sync(full,e);
    if (unsigned(e)<experts && (t&31)==__ffs(mask)-1) atomicAdd(counts+e,__popc(mask));
    __syncthreads();
    if (t<experts) grid[id*experts+t]=counts[t];
}

template<int block>
__global__ void prefix(int *grid,int tiles,int *owners,int *length) {
    __shared__ int starts[experts], sums[4];
    int e=threadIdx.x,total=0;
    for (int tile=0;tile<tiles;++tile) total+=grid[tile*experts+e];
    assign_experts<block>(total,starts,sums,owners,length);
    int p=starts[e];
    for (int tile=0;tile<tiles;++tile) {
        int *slot=grid+tile*experts+e;
        int n=*slot;
        *slot=p;
        p+=n;
    }
}

__global__ void scatter(const int *ids,int slots,const int *grid,int *out) {
    __shared__ int counts[8][experts];
    int t=threadIdx.x,w=t/32,lane=t&31;
    int pos=int(blockIdx.x)*256+t;
    for (int j=t;j<8*experts;j+=256) counts[j/experts][j%experts]=0;
    __syncthreads();
    int e=pos<slots?ids[pos]:-1;
    unsigned mask=__match_any_sync(full,e);
    if (unsigned(e)<experts && lane==__ffs(mask)-1) counts[w][e]=__popc(mask);
    __syncthreads();
    if (unsigned(e)<experts) {
        int index=grid[blockIdx.x*experts+e]+__popc(mask&((1u<<lane)-1));
        for (int j=0;j<w;++j) index+=counts[j][e];
        out[index]=pos;
    }
}

template<int block>
void run(const int *ids,int slots,int *out,int bound,int *owners,int *length,int *grid,cudaStream_t stream) {
    if (slots<=128) small<128,block><<<1,128,0,stream>>>(ids,slots,out,bound,owners,length);
    else if (slots<=256) small<256,block><<<1,256,0,stream>>>(ids,slots,out,bound,owners,length);
    else if (slots<=512) small<512,block><<<1,512,0,stream>>>(ids,slots,out,bound,owners,length);
    else if (slots<=1024) small<1024,block><<<1,1024,0,stream>>>(ids,slots,out,bound,owners,length);
    else {
        int tiles=(slots+255)/256;
        histogram<block><<<tiles+(bound+255)/256,256,0,stream>>>(ids,slots,tiles,grid,out,bound,owners);
        prefix<block><<<1,128,0,stream>>>(grid,tiles,owners,length);
        scatter<<<tiles,256,0,stream>>>(ids,slots,grid,out);
    }
}
}

extern "C" int inkling_stable_routes(const int *ids,int slots,int *out,int bound,int *owners,
                                     int *length,int *grid,int block,cudaStream_t stream) {
    if (block==8) run<8>(ids,slots,out,bound,owners,length,grid,stream);
    else if (block==16) run<16>(ids,slots,out,bound,owners,length,grid,stream);
    else if (block==48) run<48>(ids,slots,out,bound,owners,length,grid,stream);
    else return -1;
    return int(cudaPeekAtLastError());
}
