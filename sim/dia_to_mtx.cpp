/* dia_to_mtx.cpp — export a diagonal-format HamLib H to MatrixMarket .mtx
 * (coordinate real general, 1-based) so external harnesses (e.g. Drawloom)
 * can read the SAME matrix our kernels use.  usage: dia_to_mtx <dia> <out.mtx> */
#include "dia_io.hpp"
#include <cstdio>
int main(int argc,char**argv){
  if(argc<3){ std::fprintf(stderr,"usage: %s <dia_file> <out.mtx>\n",argv[0]); return 1; }
  DiaHost H=load_dia(argv[1]); CsrHost c=dia_to_csr(H);
  // Drop structural zeros so Drawloom sees the SAME true nonzeros our zero-skip
  // and cuSPARSE baselines use (dia_to_csr keeps the padded band otherwise).
  long long nz=0; for(long long p=0;p<(long long)c.vals.size();++p) if(c.vals[p]!=0.f) ++nz;
  FILE* f=std::fopen(argv[2],"w");
  std::fprintf(f,"%%%%MatrixMarket matrix coordinate real general\n");
  std::fprintf(f,"%d %d %lld\n",c.n,c.n,nz);
  for(int i=0;i<c.n;++i)
    for(int p=c.row_ptr[i];p<c.row_ptr[i+1];++p)
      if(c.vals[p]!=0.f) std::fprintf(f,"%d %d %.9g\n",i+1,c.col_idx[p]+1,c.vals[p]);
  std::fclose(f);
  std::printf("wrote %s : n=%d nnz=%lld (dropped %lld structural zeros)\n",argv[2],c.n,nz,(long long)c.nnz-nz);
  return 0;
}
