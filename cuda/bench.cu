#define main original_parity_main
#include "pulses.cu"
#undef main
#include <array>
#include <iomanip>
#include <iostream>

struct Host {
 std::vector<Sum> sums=std::vector<Sum>(R);
 std::vector<Score> scores=std::vector<Score>(R*K);
 std::vector<Byte> choices=std::vector<Byte>(R);
};
static bool host_marked(const Input&v,int i){return i<N&&v.mark[i];}
static int comp_mags(const Score&a,const Score&b){
 int i=0,j=0;
 while(true){
   while(i<P&&!a.mark[i])++i;
   while(j<P&&!b.mark[j])++j;
   if(i==P||j==P) return i==P?(j==P?0:-1):1;
   ++i;++j;
 }
}
static bool host_greater(const Score&a,const Score&b){
 if(a.sign!=b.sign)return a.sign==pos||b.sign==neg;
 if(a.sign==zero)return false;
 const int c=comp_mags(a,b);
 return a.sign==pos?c>0:c<0;
}
__attribute__((noinline)) static void host_run(const std::vector<Row>&in,Host&out){
 for(int i=0;i<R;++i){
   const Row&row=in[i];const Input&a=row.a,&b=row.b;
   Sum s{};int ai=0,bi=0,oi=0;
   if(a.sign!=b.sign&&a.sign!=zero&&b.sign!=zero)
     while(host_marked(a,ai)&&host_marked(b,bi)){++ai;++bi;}
   while(host_marked(a,ai)){s.mark[oi++]=a.mark[ai++];s.sign=a.sign;}
   while(host_marked(b,bi)){s.mark[oi++]=b.mark[bi++];s.sign=b.sign;}
   out.sums[i]=s;
 }
 for(int i=0;i<R;++i){
   const Row&row=in[i];
   for(int k=0;k<K;++k){
     Score&s=out.scores[i*K+k];
     const Input&q=row.q,&key=row.k[k];
     s.sign=(q.sign==zero||key.sign==zero)?zero:(q.sign==key.sign?pos:neg);
     for(int p=0;p<N;++p){
       // Byte copies implement the same outer-product pulses across one row; no bit packing.
       if(q.mark[p])std::memcpy(s.mark+p*N,key.mark,N);
       else std::memset(s.mark+p*N,0,N);
     }
   }
 }
 for(int i=0;i<R;++i){
   const Score*s=out.scores.data()+i*K;
   Byte best=0;
   for(Byte k=1;k<K;++k)if(host_greater(s[k],s[best]))best=k;
   out.choices[i]=best;
 }
 asm volatile("" : : "g"(out.sums.data()),"g"(out.scores.data()),"g"(out.choices.data()) : "memory");
}
struct GPU{
 Row*d=nullptr;Sum*s=nullptr;Score*qk=nullptr;Byte*idx=nullptr;
 cudaEvent_t begin,end;
 GPU(){
   csafe(cudaMalloc(&d,sizeof(Row)*R));csafe(cudaMalloc(&s,sizeof(Sum)*R));
   csafe(cudaMalloc(&qk,sizeof(Score)*R*K));csafe(cudaMalloc(&idx,sizeof(Byte)*R));
   csafe(cudaEventCreate(&begin));csafe(cudaEventCreate(&end));
 }
 void upload(const std::vector<Row>&v){csafe(cudaMemcpy(d,v.data(),sizeof(Row)*R,cudaMemcpyHostToDevice));}
 void kernels(){
   add_kernel<<<R/256,256>>>(d,s);qk_kernel<<<R*K/256,256>>>(d,qk);hardmax_kernel<<<R/256,256>>>(qk,idx);
   csafe(cudaGetLastError());
 }
 void download(Host&h){
   csafe(cudaMemcpy(h.sums.data(),s,R*sizeof(Sum),cudaMemcpyDeviceToHost));
   csafe(cudaMemcpy(h.scores.data(),qk,R*K*sizeof(Score),cudaMemcpyDeviceToHost));
   csafe(cudaMemcpy(h.choices.data(),idx,R*sizeof(Byte),cudaMemcpyDeviceToHost));
 }
 ~GPU(){
   cudaEventDestroy(begin);cudaEventDestroy(end);cudaFree(idx);cudaFree(qk);cudaFree(s);cudaFree(d);
 }
};
static void compare(const Host&a,const Host&b){
 if(std::memcmp(a.sums.data(),b.sums.data(),R*sizeof(Sum))||
 std::memcmp(a.scores.data(),b.scores.data(),R*K*sizeof(Score))||a.choices!=b.choices)
   throw std::runtime_error("complete pulse CPU/GPU parity mismatch");
}
using Wall=std::chrono::steady_clock;
int main(int argc,char**argv){
 try{
  const uint32_t seed=argc>1?std::strtoul(argv[1],nullptr,10):419071;
  // Same batch, deterministic data and fixed byte layout as original CUDA harness.
  auto in=rows(seed);Host h,g;GPU dev;
  constexpr int reps=6, rounds=9;
  const std::array<std::array<int,3>,3> order={{{0,1,2},{1,2,0},{2,0,1}}};
  host_run(in,h);dev.upload(in);dev.kernels();dev.download(g);compare(h,g);
  std::cout<<std::fixed<<std::setprecision(6)<<"{\"seed\":"<<seed<<",\"rows\":"<<R<<",\"candidates_per_row\":"<<K<<",\"reps\":"<<reps<<",\"pareto\":[";
  for(int j=0;j<rounds;++j){
    const auto& seq=order[(j+seed)%order.size()];
    std::array<double,3> values{};
    for(int arm:seq){
      if(arm==0){
        auto t=Wall::now();
        for(int i=0;i<reps;++i)host_run(in,h);
        values[arm]=std::chrono::duration<double,std::milli>(Wall::now()-t).count()/reps;
      }else if(arm==1){
        dev.upload(in);csafe(cudaDeviceSynchronize());
        csafe(cudaEventRecord(dev.begin));
        for(int i=0;i<reps;++i)dev.kernels();
        csafe(cudaEventRecord(dev.end));csafe(cudaEventSynchronize(dev.end));
        float ms=0;csafe(cudaEventElapsedTime(&ms,dev.begin,dev.end));
        values[arm]=double(ms)/reps;
        dev.download(g);
      }else{
        auto t=Wall::now();
        for(int i=0;i<reps;++i){dev.upload(in);dev.kernels();dev.download(g);}
        values[arm]=std::chrono::duration<double,std::milli>(Wall::now()-t).count()/reps;
      }
      compare(h,g); // all output bytes after each timing arm; compare excluded from timing
    }
    if(j)std::cout<<',';
    std::cout<<'['<<values[0]<<','<<values[1]<<','<<values[2]<<']';
  }
  std::cout<<"],\"fields\":[\"cpu_pulse_ms\",\"gpu_resident_ms\",\"gpu_transfer_included_ms\"],\"status\":\"PASS\"}\n";
  return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
