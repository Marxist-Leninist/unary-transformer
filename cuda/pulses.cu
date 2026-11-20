#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>
using Byte=unsigned char;
constexpr int N=16, P=N*N, K=8, R=4096, L=64;
constexpr Byte zero=0, pos=1, neg=2; // categorical sign tags, not magnitudes
struct Input{Byte sign, mark[N];};
struct Sum{Byte sign, mark[N+N];};
struct Score{Byte sign, mark[P];};
struct Row{Input a,b,q,k[K];};
template<typename U> void csafe(U e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ bool marked(const Input&v,int ix){return ix<N && v.mark[ix];}
__global__ void add_kernel(const Row*in,Sum*out){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=R)return;
    const Input &a=in[row].a,&b=in[row].b;
    Sum s{};int ai=0,bi=0,oi=0; // only pulse-array cursors
    if(a.sign!=b.sign && a.sign!=zero && b.sign!=zero)
        while(marked(a,ai)&&marked(b,bi)){++ai;++bi;} // cancel matched opposite pulses
    while(marked(a,ai)){s.mark[oi++]=a.mark[ai++];s.sign=a.sign;}
    while(marked(b,bi)){s.mark[oi++]=b.mark[bi++];s.sign=b.sign;}
    out[row]=s;
}
__global__ void qk_kernel(const Row*in,Score*out){
    const int record=blockIdx.x*blockDim.x+threadIdx.x;
    if(record>=R*K)return;
    const Row &r=in[record/K];
    const auto &a=r.q,&b=r.k[record%K];
    auto &s=out[record]; 
    s.sign=(a.sign==zero||b.sign==zero)?zero:(a.sign==b.sign?pos:neg);
    // Literal tensor outer product: one slot for every pair, no scalar product decode.
    for(int i=0;i<N;++i)
        for(int j=0;j<N;++j)
            s.mark[i*N+j]=(a.mark[i] && b.mark[j]);
}
__device__ int mag_compare(const Score&a,const Score&b){
    int i=0,j=0; // array cursor only; never count marks
    while(true){
        while(i<P && !a.mark[i])++i;
        while(j<P && !b.mark[j])++j;
        if(i==P || j==P) return i==P ? (j==P?0:-1):1; // ordering label
        ++i;++j; // discard one matched pair, not a scalar sum
    }
}
__device__ bool greater(const Score&a,const Score&b){
    if(a.sign!=b.sign){
        if(a.sign==pos || b.sign==neg)return true;
        return false;
    }
    if(a.sign==zero)return false;
    const int comparison=mag_compare(a,b);
    return a.sign==pos?comparison>0:comparison<0;
}
__global__ void hardmax_kernel(const Score*in,Byte*out){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=R)return;
    const Score *choices=in+row*K;
    Byte best=0;
    for(Byte k=1;k<K;++k)if(greater(choices[k],choices[best]))best=k;
    out[row]=best; // categorical selected key, not a magnitude
}
static std::string tally(const Input& x){
    std::string out;
    for(Byte m:x.mark) if(m)out+= x.sign==neg?'-':'+';
    return out;
}
static std::string ref_add(const std::string&x,const std::string&y){
    auto out=x;
    for(char c:y)
        if(!out.empty()&&out.back()!=c)out.pop_back();
        else out.push_back(c);
    return out;
}
static std::string ref_mul(const std::string&x,const std::string&y){
    if(x.empty()||y.empty())return {};
    std::string out;
    for(char a:x)for(char b:y)out.push_back(a==b?'+':'-');
    return out;
}
static bool ref_greater(const std::string&a,const std::string&b){
    const auto count=ref_add(a,std::string(b.size(),b.empty()||b.front()=='-'?'+':'-'));
    return !count.empty()&&count.front()=='+';
}
static Input fixture(int v){
    Input x{};
    x.sign=v==0?zero:v<0?neg:pos;
    for(int i=0;i<std::abs(v);++i)x.mark[i]=1;
    return x;
}
static std::vector<Row> rows(uint32_t seed){
    std::mt19937 rng(seed);std::uniform_int_distribution<int> uni(-N,N);
    std::vector<Row> rs(R);
    int ai=-N,bi=-N;
    for(auto &r:rs){
        r.a=fixture(ai<=N?ai:uni(rng));r.b=fixture(ai<=N?bi:uni(rng));r.q=fixture(uni(rng));
        if(bi==N){++ai;bi=-N;}else ++bi;
        for(auto &k:r.k)k=fixture(uni(rng));
    }
    return rs;
}
int main(int argc,char**argv){
    try{
        const uint32_t seed=argc>1?std::strtoul(argv[1],nullptr,10):419071;
        auto inputs=rows(seed);
        Row*d=nullptr;Sum*s=nullptr;Score*qk=nullptr;Byte*idx=nullptr;
        std::vector<Sum> adds(R);std::vector<Score> scores(R*K);std::vector<Byte> choices(R);
        size_t free_before,total,free_after;
        csafe(cudaFree(nullptr));csafe(cudaMemGetInfo(&free_before,&total));
        const auto start=std::chrono::steady_clock::now();
        csafe(cudaMalloc(&d,sizeof(Row)*R));csafe(cudaMalloc(&s,sizeof(Sum)*R));
        csafe(cudaMalloc(&qk,sizeof(Score)*R*K));csafe(cudaMalloc(&idx,sizeof(Byte)*R));
        csafe(cudaMemGetInfo(&free_after,&total));
        csafe(cudaMemcpy(d,inputs.data(),sizeof(Row)*R,cudaMemcpyHostToDevice));
        auto fire=[&](){
            add_kernel<<<R/256,256>>>(d,s);qk_kernel<<<R*K/256,256>>>(d,qk);hardmax_kernel<<<R/256,256>>>(qk,idx);
            csafe(cudaGetLastError());
        };
        fire();csafe(cudaDeviceSynchronize());
        cudaEvent_t begin,end;csafe(cudaEventCreate(&begin));csafe(cudaEventCreate(&end));
        csafe(cudaEventRecord(begin));
        for(int rep=0;rep<L;++rep)fire();
        csafe(cudaEventRecord(end));csafe(cudaEventSynchronize(end));
        float milliseconds;csafe(cudaEventElapsedTime(&milliseconds,begin,end)); // telemetry only
        csafe(cudaMemcpy(adds.data(),s,R*sizeof(Sum),cudaMemcpyDeviceToHost));
        csafe(cudaMemcpy(scores.data(),qk,R*K*sizeof(Score),cudaMemcpyDeviceToHost));
        csafe(cudaMemcpy(choices.data(),idx,R*sizeof(Byte),cudaMemcpyDeviceToHost));
        const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        for(int r=0;r<R;++r){
            const Row &v=inputs[r];const auto &u=adds[r];
            const auto expected=ref_add(tally(v.a),tally(v.b));
            const auto sign=expected.empty()?zero:expected.front()=='-'?neg:pos;
            if(sign!=u.sign)throw std::runtime_error("signed add sign parity");
            for(int slot=0;slot<N+N;++slot)
                if((slot<(int)expected.size())!=u.mark[slot])throw std::runtime_error("signed add pulse parity");
            int want=0;std::string best;
            for(int k=0;k<K;++k){
                const auto &p=scores[r*K+k];
                const auto y=ref_mul(tally(v.q),tally(v.k[k]));
                const auto ys=y.empty()?zero:y.front()=='-'?neg:pos;
                if(ys!=p.sign)throw std::runtime_error("QK sign parity");
                std::string decoded;
                for(int i=0;i<N;++i)for(int j=0;j<N;++j){
                    const bool should=v.q.mark[i]&&v.k[k].mark[j];
                    if(should!=p.mark[i*N+j])throw std::runtime_error("QK slot parity");
                    if(p.mark[i*N+j])decoded+=p.sign==neg?'-':'+';
                }
                if(decoded!=y)throw std::runtime_error("QK decoded parity");
                if(k==0||ref_greater(y,best)){best=y;want=k;}
            }
            if(want!=choices[r])throw std::runtime_error("hardmax first-tie parity");
        }
        cudaDeviceProp prop;csafe(cudaGetDeviceProperties(&prop,0));
        std::printf("{\"seed\":%u,\"gpu\":\"%s\",\"sm\":\"%d.%d\",\"exact_add\":%d,\"exact_qk\":%d,\"exact_argmax_rows\":%d,\"kernel_triples\":%d,\"events_ms\":%.5f,\"own_buffers_bytes\":%zu,\"device_free_delta_bytes\":%zu,\"allocator_through_copies_wall_ms\":%.5f,\"status\":\"PASS\"}\n",
          seed,prop.name,prop.major,prop.minor,R,R*K,R,L,milliseconds,sizeof(Row)*R+sizeof(Sum)*R+sizeof(Score)*R*K+sizeof(Byte)*R,free_before-free_after,wall);
        csafe(cudaFree(idx));csafe(cudaFree(qk));csafe(cudaFree(s));csafe(cudaFree(d));
        return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
