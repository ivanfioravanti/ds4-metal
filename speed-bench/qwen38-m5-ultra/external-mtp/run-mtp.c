/* Separate original-versus-candidate same-depth MTP provenance launcher.
 * prepare-launcher.sh generates launcher_config.h after both diagnostic builds.
 * No runtime, benchmark protocol, model or original launcher is modified. */
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "launcher_config.h"
static const struct {const char *key,*relative;} metal_sources[]={
#include "runner_metal_paths.inc"
};
static void die(const char *s){fprintf(stderr,"external-mtp-runner: %s (%s)\n",s,strerror(errno));exit(1);}
static void joined(char out[PATH_MAX],const char *a,const char *b){if(snprintf(out,PATH_MAX,"%s/%s",a,b)>=PATH_MAX){errno=ENAMETOOLONG;die(b);}}
static void hex(const unsigned char *bytes,char out[65]){for(int i=0;i<32;i++)snprintf(out+2*i,3,"%02x",bytes[i]);}
static void file_hash(const char *path,char out[65]){
    FILE *f=fopen(path,"rb");if(!f)die(path);unsigned char buf[65536],hash[32];size_t n;
    CC_SHA256_CTX s;CC_SHA256_Init(&s);while((n=fread(buf,1,sizeof(buf),f)))CC_SHA256_Update(&s,buf,(CC_LONG)n);
    if(ferror(f))die(path);fclose(f);CC_SHA256_Final(hash,&s);hex(hash,out);
}
static void require_hash(const char *path,const char *expected){char hash[65];file_hash(path,hash);if(strcmp(hash,expected)){errno=EINVAL;die(path);}}
static void json(const char *s){putchar('"');for(const unsigned char *p=(const unsigned char*)s;*p;p++){
    if(*p=='"'||*p=='\\'){putchar('\\');putchar(*p);}else if(*p<32)printf("\\u%04x",*p);else putchar(*p);}putchar('"');}
static char *absolute(const char *cwd,const char *s){if(*s=='/')return strdup(s);char path[PATH_MAX];joined(path,cwd,s);return strdup(path);}
static void build_manifest(const char *root,const char *build,const char *expected){
    char path[PATH_MAX],line[PATH_MAX+68],actual[65];joined(path,build,"runtime.sha256");require_hash(path,expected);
    FILE *f=fopen(path,"r");if(!f)die(path);size_t rows=0;
    while(fgets(line,sizeof(line),f)){
        size_t n=strlen(line);if(n<68||line[64]!=' '||line[65]!=' '||line[n-1]!='\n'){errno=EINVAL;die("malformed build runtime manifest");}
        line[n-1]=0;const char *relative=line+66;
        if(*relative=='/'||strstr(relative,"..")){errno=EINVAL;die("unsafe build runtime path");}
        joined(path,root,relative);file_hash(path,actual);if(strncmp(actual,line,64)){errno=EINVAL;die("diagnostic build inputs changed");}rows++;
    }
    if(ferror(f)||rows<40){errno=EINVAL;die("incomplete build manifest");}fclose(f);
}
int main(int argc,char **argv){
    int at=1;bool candidate=at<argc&&!strcmp(argv[at],"--candidate");if(candidate)at++;
    bool verify=at<argc&&!strcmp(argv[at],"--verify-only");if(verify)at++;
    if(at>=argc||strcmp(argv[at++],"mtp")){fprintf(stderr,"usage: %s [--candidate] [--verify-only] mtp BENCHMARK_OPTIONS --output NEW_LOG\n",argv[0]);return 2;}
    const char *root=candidate?CANDIDATE_ROOT:ORIGINAL_ROOT,*build=candidate?CANDIDATE_BUILD:ORIGINAL_BUILD;
    char cwd[PATH_MAX],path[PATH_MAX],binary[PATH_MAX],runtime_hash[65],hash[65];if(!getcwd(cwd,sizeof(cwd)))die("cwd");
    char **args=calloc((size_t)argc+1,sizeof(*args));if(!args)die("argv");joined(binary,build,"metal_decode_mtp_same_depth");args[0]=binary;int count=1;char *output=NULL;
    for(;at<argc;at++){
        bool out=!strcmp(argv[at],"--output"),input=!strcmp(argv[at],"--model")||!strcmp(argv[at],"-m")||!strcmp(argv[at],"--prompt-file");
        if(out||input){if(at+1>=argc){errno=EINVAL;die("missing path");}char *value=absolute(cwd,argv[at+1]);if(!value)die("path allocation");
            if(out){if(output){errno=EINVAL;die("duplicate output");}output=value;}else{args[count++]=argv[at];args[count++]=value;}at++;
        }else args[count++]=argv[at];
    }
    if(!verify&&!output){errno=EINVAL;die("explicit new output required");}
    if(output){int fd=open(output,O_WRONLY|O_CREAT|O_EXCL,0644);if(fd<0||dup2(fd,1)<0||dup2(fd,2)<0)die("new output");close(fd);}
    require_hash(binary,candidate?CANDIDATE_BINARY_SHA256:ORIGINAL_BINARY_SHA256);
    build_manifest(root,build,candidate?CANDIDATE_BUILD_RUNTIME_SHA256:ORIGINAL_BUILD_RUNTIME_SHA256);
    const char *names[]={"metal_decode_mtp_same_depth.c","metal_decode_schedule_bench.c"};
    const char *expected[]={MTP_WRAPPER_SHA256,STRICT_PROTOCOL_SHA256};
    for(unsigned i=0;i<2;i++){joined(path,build,names[i]);require_hash(path,expected[i]);}
    joined(path,root,"speed-bench/metal_decode_schedule_bench.c");require_hash(path,STRICT_PROTOCOL_SHA256);
    joined(path,root,"speed-bench/benchmark_pool.h");require_hash(path,POOL_H_SHA256);
    joined(path,root,"speed-bench/benchmark_pool.m");require_hash(path,POOL_M_SHA256);
    joined(path,ORIGINAL_ROOT,"ORIGINAL_RUNTIME_SHA256.txt");require_hash(path,ORIGINAL_MANIFEST_SHA256);
    FILE *f=fopen(path,"r");if(!f)die(path);char line[PATH_MAX+68];size_t files=0;CC_SHA256_CTX total;CC_SHA256_Init(&total);
    while(fgets(line,sizeof(line),f)){
        size_t n=strlen(line);if(n<68||line[64]!=' '||line[65]!=' '||line[n-1]!='\n'){errno=EINVAL;die("original inventory");}
        line[n-1]=0;const char *relative=line+66;if(*relative=='/'||strstr(relative,"..")){errno=EINVAL;die("runtime path");}
        joined(path,root,relative);file_hash(path,hash);if(!candidate&&strncmp(hash,line,64)){errno=EINVAL;die("original runtime changed");}
        CC_SHA256_Update(&total,relative,(CC_LONG)strlen(relative));CC_SHA256_Update(&total,hash,64);files++;
    }
    if(ferror(f))die("runtime inventory read");fclose(f);unsigned char total_bytes[32];CC_SHA256_Final(total_bytes,&total);hex(total_bytes,runtime_hash);
    if(files!=167||(!candidate&&strcmp(runtime_hash,"822fe7e1d0491f5b4a830147e79d2724667a576ce9f430bfc2da02eb392bd17d"))){errno=EINVAL;die("runtime ancestry/inventory mismatch");}
    for(size_t i=0;i<sizeof(metal_sources)/sizeof(metal_sources[0]);i++){joined(path,root,metal_sources[i].relative);if(setenv(metal_sources[i].key,path,1))die("pin Metal");}
    printf("build_provenance={\"format\":\"external-mtp-v1\",\"runtime\":\"%s\",\"baseline_kind\":\"original\",\"original_revision\":\"0aaea5a238fb41a35106a551e73c8409dfb751ac\",\"verification_only\":%s,\"runtime_files\":167,\"runtime_files_sha256\":\"%s\",\"binary_sha256\":\"%s\",\"wrapper_sha256\":\"%s\",\"strict_protocol_sha256\":\"%s\",\"pool_h_sha256\":\"%s\",\"pool_m_sha256\":\"%s\",\"metal_sources_pinned\":26,\"cwd\":",
        candidate?"candidate":"original",verify?"true":"false",runtime_hash,candidate?CANDIDATE_BINARY_SHA256:ORIGINAL_BINARY_SHA256,MTP_WRAPPER_SHA256,STRICT_PROTOCOL_SHA256,POOL_H_SHA256,POOL_M_SHA256);
    json(root);printf(",\"program\":");json(binary);printf(",\"command_args\":[");for(int i=1;i<count;i++){if(i>1)putchar(',');json(args[i]);}printf("]}\n");fflush(NULL);
    if(verify)return 0;if(chdir(root))die("runtime cwd");execv(binary,args);die("exec diagnostic");
}
