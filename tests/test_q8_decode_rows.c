#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require(int ok) { if (!ok) { fprintf(stderr, "Q8 decode rows test failed\n"); exit(1); } }
static uint32_t seed=123;
static void *maps[32];
static unsigned map_count;
static uint32_t rnd(void) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; return seed; }
static void check(uint32_t d, uint32_t o, uint32_t t) {
    const uint64_t bytes=(uint64_t)d/32*34*o, n=(uint64_t)t*o;
    const uint64_t page=sysconf(_SC_PAGESIZE), stride=(bytes+page-1)/page*page;
    void *map=NULL;
    require(posix_memalign(&map,page,2*stride)==0);
    maps[map_count++]=map;
    memset(map,0,2*stride);
    for (uint32_t matrix=0; matrix<2; matrix++) for (uint64_t b=0; b<bytes; b+=34) {
        uint8_t *p=(uint8_t *)map+matrix*stride+b;
        uint16_t scale=0x1800+(rnd()%16)*0x100;
        memcpy(p,&scale,2);
        for (int j=2; j<34; j++) p[j]=rnd();
    }
    require(ds4_gpu_set_model_map(map,2*stride));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((uint64_t)t*d*4);
    ds4_gpu_tensor *a=ds4_gpu_tensor_alloc((n+16)*4), *b=ds4_gpu_tensor_alloc((n+16)*4);
    require(x && a && b);
    float *input=malloc((uint64_t)t*d*4), *actual=malloc((n+16)*4);
    float *refs[2]={malloc(n*4),malloc(n*4)};
    require(input && actual && refs[0] && refs[1]);
    for (uint64_t i=0; i<(uint64_t)t*d; i++) input[i]=((int)(rnd()%257)-128)/256.f;
    require(ds4_gpu_tensor_write(x,0,input,(uint64_t)t*d*4));
    const char *variants[]={"1",NULL,"1",NULL,"1"};
    for (int run=0; run<5; run++) {
        if (variants[run]) require(setenv("DS4_METAL_DISABLE_Q8_TOKEN_PAIR",variants[run],1)==0);
        else require(unsetenv("DS4_METAL_DISABLE_Q8_TOKEN_PAIR")==0);
        require(ds4_gpu_tensor_fill_f32(a,NAN,n+16) && ds4_gpu_tensor_fill_f32(b,NAN,n+16));
        require(ds4_gpu_begin_commands());
        require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(a,map,2*stride,0,d,o,x,t));
        require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(b,map,2*stride,stride,d,o,x,t));
        require(ds4_gpu_end_commands());
        ds4_gpu_tensor *outputs[]={a,b};
        for (int m=0; m<2; m++) {
            require(ds4_gpu_tensor_read(outputs[m],0,actual,(n+16)*4));
            for (uint64_t i=0; i<n; i++) require(isfinite(actual[i]));
            for (uint64_t i=n; i<n+16; i++) require(isnan(actual[i]));
            if (!run) memcpy(refs[m],actual,n*4);
            else if (memcmp(refs[m],actual,n*4)) {
                fprintf(stderr,"Mismatch D=%u O=%u T=%u variant=%s matrix=%d\n",d,o,t,variants[run] ? variants[run] : "default",m);
                exit(1);
            }
        }
    }
    /* Compare fused rounding with the original exact matvec followed by
     * the model's BF16 boundary, including odd token counts and guards. */
    for (int paired=0; paired<2; paired++) {
        if (paired) unsetenv("DS4_METAL_DISABLE_Q8_TOKEN_PAIR");
        else setenv("DS4_METAL_DISABLE_Q8_TOKEN_PAIR","1",1);
        require(ds4_gpu_tensor_fill_f32(a,NAN,n+16) && ds4_gpu_tensor_fill_f32(b,NAN,n+16));
        require(ds4_gpu_begin_commands());
        require(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(a,map,2*stride,0,d,o,x,t));
        require(ds4_gpu_dsv41_quantize(a,o,t,DS4_V41_BF16));
        require(ds4_gpu_dsv41_q8_rows_bf16(b,map,2*stride,0,d,o,x,t));
        require(ds4_gpu_end_commands());
        require(ds4_gpu_tensor_read(a,0,refs[0],n*4));
        require(ds4_gpu_tensor_read(b,0,actual,(n+16)*4));
        require(memcmp(refs[0],actual,n*4)==0);
        for (uint64_t i=n;i<n+16;i++) require(isnan(actual[i]));
    }
    printf("PASS Q8 D=%u O=%u T=%u exact paired outputs, guards intact\n",d,o,t);
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(x);
    free(input); free(actual); free(refs[0]); free(refs[1]);
}
int main(void) {
    require(ds4_gpu_init());
    const uint32_t shapes[][3]={{32,1,1},{32,1,2},{64,48,3},{96,63,4},{128,65,5},
        {5120,512,2},{5120,512,3},{5120,512,4},{5120,512,5},{5120,512,6},
        {1024,8192,2},{5120,65568,2},{5120,65568,3},{2304,5120,2},{128,65,7}};
    for (unsigned i=0; i<sizeof(shapes)/sizeof(*shapes); i++) check(shapes[i][0],shapes[i][1],shapes[i][2]);
    unsetenv("DS4_METAL_DISABLE_Q8_TOKEN_PAIR");
    ds4_gpu_cleanup();
    for (unsigned i=0; i<map_count; i++) free(maps[i]);
    return 0;
}
