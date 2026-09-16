/* Exact batched DSpark attention against the scalar kernel, including
 * causal ring staging and the drafter's shared noncausal keys. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
static uint32_t state=314159;
static float value(void) {state^=state<<13;state^=state>>17;state^=state<<5;return ((int)(state%20001)-10000)/32768.f;}
static ds4_gpu_tensor *floats(size_t n) {
 ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(n*4);CHECK(t);
 float *x=ds4_gpu_tensor_contents(t);CHECK(x);
 for(size_t i=0;i<n;i++)x[i]=value();return t;
}
static int verify_cases(void *map) {
 for(uint32_t heads=32;heads<=64;heads+=32)for(uint32_t nc=0;nc<=512;nc+=512)for(uint32_t rows=2;rows<=6;rows++){
  size_t n=(size_t)rows*heads*512;
  ds4_gpu_tensor *q=floats(n),*out=floats(n+16),*ref=floats(n+16),*comp=floats(1024*512);
  ds4_gpu_tensor *staged=ds4_gpu_tensor_alloc((uint64_t)6*640*512*2);CHECK(staged);
  ds4_gpu_tensor *raw[6]={0},*ids[6]={0};
  for(uint32_t i=0;i<rows;i++) {
   raw[i]=floats(128*512);ids[i]=ds4_gpu_tensor_alloc(512*4);CHECK(ids[i]);
   int32_t *ip=ds4_gpu_tensor_contents(ids[i]);for(int j=0;j<512;j++)ip[j]=(j*7+i*31)%1024;
  }
  CHECK(ds4_gpu_tensor_fill_f32(out,NAN,n+16));CHECK(ds4_gpu_tensor_fill_f32(ref,NAN,n+16));
  ds4_gpu_tensor *poison=floats(128*512);CHECK(ds4_gpu_tensor_fill_f32(poison,NAN,128*512));
  CHECK(ds4_gpu_begin_commands());
  for(uint32_t i=0;i<rows;i++) {
   uint32_t start=(126+i)%128;
   ds4_gpu_tensor *qv=ds4_gpu_tensor_view(q,(uint64_t)i*heads*512*4,(uint64_t)heads*512*4);
   ds4_gpu_tensor *rv=ds4_gpu_tensor_view(ref,(uint64_t)i*heads*512*4,(uint64_t)heads*512*4);
   CHECK(qv&&rv);
   if(nc)CHECK(ds4_gpu_dsv41_attention_selected(rv,map,getpagesize(),0,qv,raw[i],128,128,start,comp,nc,heads,512,ids[i]));
   else CHECK(ds4_gpu_attention_decode_heads_tensor(rv,map,getpagesize(),0,qv,raw[i],128,128,start,NULL,0,0,NULL,0,heads,512));
   CHECK(ds4_gpu_dsv41_verify_attention_stage(staged,raw[i],start,comp,ids[i],nc,i));
   CHECK(ds4_gpu_tensor_copy(raw[i],0,poison,0,128*512*4));
   ds4_gpu_tensor_free(qv);ds4_gpu_tensor_free(rv);
  }
  CHECK(ds4_gpu_dsv41_verify_attention(out,map,getpagesize(),0,q,staged,nc,heads,rows));
  CHECK(ds4_gpu_end_commands());
  float *a=ds4_gpu_tensor_contents(out),*b=ds4_gpu_tensor_contents(ref);
  for(size_t i=0;i<n;i++)CHECK(isfinite(a[i]));
  if(memcmp(a,b,n*4)) {for(size_t i=0;i<n;i++)if(memcmp(a+i,b+i,4)){fprintf(stderr,"heads%u keys%u rows%u idx%zu %.9g %.9g\n",heads,nc+128,rows,i,a[i],b[i]);break;}return 1;}
  for(size_t i=n;i<n+16;i++)CHECK(isnan(a[i]));
  printf("PASS heads=%u keys=%u rows=%u exact, ring-wrap stage isolated, guards intact\n",heads,nc+128,rows);
  ds4_gpu_tensor_free(poison);ds4_gpu_tensor_free(q);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ref);ds4_gpu_tensor_free(comp);ds4_gpu_tensor_free(staged);
  for(uint32_t i=0;i<rows;i++){ds4_gpu_tensor_free(raw[i]);ds4_gpu_tensor_free(ids[i]);}
 }
 return 0;
}

static int draft_cases(void *map) {
 const uint32_t lengths[]={1,18,27,28,31,32,123,124,127,128};
 for(size_t c=0;c<sizeof(lengths)/sizeof(*lengths);c++){
  uint32_t raw_count=lengths[c];size_t n=5*64*512;
  ds4_gpu_tensor *q=floats(n),*out=floats(n+16),*ref=floats(n+16);
  ds4_gpu_tensor *raw=floats(128*512),*kv=floats(5*512);
  CHECK(ds4_gpu_tensor_fill_f32(out,NAN,n+16));CHECK(ds4_gpu_tensor_fill_f32(ref,NAN,n+16));
  CHECK(ds4_gpu_begin_commands());
  for(uint32_t i=0;i<5;i++) {
   ds4_gpu_tensor *qv=ds4_gpu_tensor_view(q,(uint64_t)i*64*512*4,64*512*4);
   ds4_gpu_tensor *rv=ds4_gpu_tensor_view(ref,(uint64_t)i*64*512*4,64*512*4);
   CHECK(qv&&rv);
   CHECK(ds4_gpu_attention_decode_heads_tensor(rv,map,getpagesize(),0,qv,raw,raw_count,128,0,kv,0,5,NULL,0,64,512));
   ds4_gpu_tensor_free(qv);ds4_gpu_tensor_free(rv);
  }
  CHECK(ds4_gpu_dsv41_draft_attention(out,map,getpagesize(),0,q,raw,raw_count,kv));
  CHECK(ds4_gpu_end_commands());
  float *a=ds4_gpu_tensor_contents(out),*b=ds4_gpu_tensor_contents(ref);
  for(size_t i=0;i<n;i++)CHECK(isfinite(a[i]));
  if(memcmp(a,b,n*4)){for(size_t i=0;i<n;i++)if(memcmp(a+i,b+i,4)){fprintf(stderr,"raw%u idx%zu %.9g %.9g\n",raw_count,i,a[i],b[i]);break;}return 1;}
  for(size_t i=n;i<n+16;i++)CHECK(isnan(a[i]));
  printf("PASS draft keys=%u rows=5 exact, guards intact\n",raw_count+5);
  ds4_gpu_tensor_free(q);ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(ref);ds4_gpu_tensor_free(raw);ds4_gpu_tensor_free(kv);
 }
 return 0;
}

int main(void) {
 CHECK(ds4_gpu_init());
 void *map=0;CHECK(!posix_memalign(&map,getpagesize(),getpagesize()));memset(map,0,getpagesize());
 for(int i=0;i<64;i++)((float*)map)[i]=value();
 CHECK(ds4_gpu_set_model_map(map,getpagesize()));
 CHECK(verify_cases(map)==0);
 CHECK(draft_cases(map)==0);
 ds4_gpu_cleanup();free(map);return 0;
}
