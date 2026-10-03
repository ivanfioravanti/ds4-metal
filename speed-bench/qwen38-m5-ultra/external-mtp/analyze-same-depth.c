/* Native paired same-depth MTP analyzer. No inference or acceptance decision.
 * Single-run logs require a separate cross-process adapter and are refused. */
#define _POSIX_C_SOURCE 200809L
#include <CommonCrypto/CommonDigest.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { BLOCKS=256, RUNS=BLOCKS*4, FRONTIERS=129, FILES=64, PATHCAP=4096 };
typedef struct { int offset,n,id; unsigned long long bytes; char state[65],logits[65]; } frontier;
typedef struct {
    int variant,cycles,cycle_rows,offset,resources,memory,hash;
    int process_valid,swap_valid,vm_valid;
    double seconds,first_ms,cycle_sum;
    unsigned long long resident,footprint,reads,writes,swap,swapins,swapouts;
} sample;
typedef struct {
    const char *path; char config_line[16384],model[PATHCAP],prompt[PATHCAP],env[128],control[32],candidate[32];
    int errors,valid,config,single,prefix,ctx,warmup,blocks,vocab,layers,depth;
    int reference,reference_cycles,warm_cycles,continuation_cycles,nrefs,refhash;
    int selections,agreements,drifts,evidence,declared_agreements,declared_frontiers;
    int drift_offsets[FRONTIERS],trajectory_ids[FRONTIERS],validation[2],runs,paired,summary,aggregate[2];
    int n[2];double seconds[2],first_ms[2],paired_printed[BLOCKS],summary_values[3],gain,low,high;
    char trajectory[65],plan[65];frontier refs[FRONTIERS];sample samples[RUNS];
} report;
static const double critical[]={0,12.706205,4.302653,3.182446,2.776445,2.570582,
    2.446912,2.364624,2.306004,2.262157,2.228139,2.200985,2.178813,2.160369,
    2.144787,2.131450,2.119906,2.109816,2.100922,2.093024,2.085964,2.079614,
    2.073874,2.068658,2.063899,2.059539,2.055530,2.051831,2.048407,2.045230,2.042273};

static int field(const char *line,const char *key,char *out,size_t cap) {
    char needle[128];if(snprintf(needle,sizeof(needle),"%s=",key)>=(int)sizeof(needle))return 0;
    const char *p=line;int found=0;
    while((p=strstr(p,needle))) {
        if(p!=line&&!isspace((unsigned char)p[-1])){p++;continue;}
        p+=strlen(needle);size_t n=strcspn(p," \t\r\n");
        if(found||!n||n>=cap)return 0;
        memcpy(out,p,n);out[n]=0;found=1;p+=n;
    }
    return found;
}
static int is(const char *line,const char *key,const char *value) {
    char text[PATHCAP];return field(line,key,text,sizeof(text))&&!strcmp(text,value);
}
static int uint_field(const char *line,const char *key,unsigned long long *out) {
    char text[64],*end;if(!field(line,key,text,sizeof(text)))return 0;
    for(const char *p=text;*p;p++)if(!isdigit((unsigned char)*p))return 0;
    errno=0;*out=strtoull(text,&end,10);return !errno&&!*end;
}
static int integer(const char *line,const char *key) {
    unsigned long long x;return uint_field(line,key,&x)&&x<=INT32_MAX?(int)x:-1;
}
static int finite_number(const char *line,const char *key,double *out) {
    char text[64],*end;if(!field(line,key,text,sizeof(text)))return 0;
    errno=0;*out=strtod(text,&end);return !errno&&!*end&&isfinite(*out);
}
static int number(const char *line,const char *key,double *out) {return finite_number(line,key,out)&&*out>=0;}
static int valid_hash(const char *p) {
    if(strlen(p)!=64)return 0;
    for(int i=0;i<64;i++)if(!isdigit((unsigned char)p[i])&&(p[i]<'a'||p[i]>'f'))return 0;
    return 1;
}
static int digest(const char *line,const char *key,char out[65]) {return field(line,key,out,65)&&valid_hash(out);}
static int variant(const char *line) {return is(line,"variant","control")?0:is(line,"variant","candidate")?1:-1;}
static int mode(const char *line) {return is(line,"mode","mtp_same_depth");}
static int pending(const char *line) {return is(line,"correctness","integration_pending")&&is(line,"quality","pending");}
static int near(double x,double y,double tolerance) {return fabs(x-y)<=tolerance;}
static int rate_matches(double seconds,double rate,unsigned tokens) {
    if(seconds<=0||rate<=0)return 0;
    return rate+0.00000051 >= tokens/(seconds+0.00000000051) &&
        (seconds<=0.00000000051 || rate-0.00000051<=tokens/(seconds-0.00000000051));
}
static void sha_finish(CC_SHA256_CTX *ctx,char hex[65]) {
    unsigned char out[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(out,ctx);
    for(unsigned i=0;i<sizeof(out);i++)snprintf(hex+2*i,3,"%02x",out[i]);
}
static void sha_bytes(const void *data,size_t size,char hex[65]) {
    CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);const unsigned char *p=data;
    while(size){CC_LONG n=size>UINT32_MAX?UINT32_MAX:(CC_LONG)size;CC_SHA256_Update(&ctx,p,n);p+=n;size-=n;}
    sha_finish(&ctx,hex);
}
static int sha_file(const char *path,char hex[65]) {
    FILE *f=fopen(path,"rb");if(!f)return 0;CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);
    unsigned char buf[65536];size_t n;while((n=fread(buf,1,sizeof(buf),f)))CC_SHA256_Update(&ctx,buf,(CC_LONG)n);
    int okay=!ferror(f);fclose(f);sha_finish(&ctx,hex);return okay;
}
static void plan_hash(const report *r,char hex[65]) {
    unsigned char bytes[FRONTIERS*12];
    for(int i=0;i<r->nrefs;i++) {
        const uint32_t v[]={(uint32_t)r->refs[i].offset,(uint32_t)r->refs[i].n,(uint32_t)r->refs[i].id};
        for(unsigned j=0;j<3;j++)for(unsigned k=0;k<4;k++)bytes[i*12+j*4+k]=(unsigned char)(v[j]>>(k*8));
    }
    sha_bytes(bytes,(size_t)r->nrefs*12,hex);
}
static int allowed(const char *name,const char *value) {
    static const struct {const char *name,*values;} controls[]={
        {"DS4_QWEN4_MTP_GPU_ARGMAX","0,1"},{"DS4_QWEN4_MTP_SWAP_RESTORE","0,1"},
        {"DS4_QWEN4_FLUSH_LAYER","0,1,2,4,8"},{"DS4_QWEN4_Q4K_MID_NSG","1,2,4,8"},
        {"DS4_QWEN4_Q4K_MID_NR","1,2"},{"DS4_QWEN4_MOE_MV_NSG","4,8,16"},
        {"DS4_QWEN4_MOE_MV_NR","1,2,4"},{"DS4_QWEN4_MOE_DOWN_PREFETCH","0,1"},
        {"DS4_METAL_PLAIN_MV_NR0","0,2"},{"DS4_QWEN4_DECODE_FUSIONS","0,1"},
        {"DS4_QWEN4_ATTN_QUERY_GROUP","4,6,12"},{"DS4_QWEN4_ATTN_MERGE_WIDE","0,1"},
        {"DS4_QWEN4_GDN_NSG","1,2,4,8"},{"DS4_QWEN4_GDN_FRONT_THREADS","256,512,1024"},
        {"DS4_QWEN4_NGRAM_OVERLAP","0,1"}};
    for(size_t i=0;i<sizeof(controls)/sizeof(controls[0]);i++)if(!strcmp(name,controls[i].name))
        for(const char *p=controls[i].values;*p;){const char *end=strchr(p,',');size_t n=end?(size_t)(end-p):strlen(p);
            if(strlen(value)==n&&!memcmp(p,value,n))return 1;if(!end)break;p=end+1;}
    return 0;
}

static void consume(report *r,const char *line) {
    if(!strncmp(line,"configuration ",14)) {
        if(++r->config!=1)return (void)r->errors++;
        snprintf(r->config_line,sizeof(r->config_line),"%s",line);r->config_line[strcspn(r->config_line,"\r\n")]=0;
        r->prefix=integer(line,"prefix");r->ctx=integer(line,"ctx");r->warmup=integer(line,"warmup");
        r->blocks=integer(line,"repeats");r->vocab=integer(line,"vocab");r->layers=integer(line,"layers");r->depth=integer(line,"mtp_depth");
        r->single=is(line,"single_run","yes");
        if(!mode(line)||!is(line,"single_run","no")||!is(line,"include_selection","yes")||
           !is(line,"reference","control_mtp_same_depth")||!is(line,"drafts","full_vocabulary")||
           !is(line,"acceptance","production")||!is(line,"unchanged_strict_harness","yes")||
           !is(line,"control","2/32")||!is(line,"candidate","2/32")||
           !field(line,"model",r->model,sizeof(r->model))||!field(line,"prompt",r->prompt,sizeof(r->prompt))||
           !field(line,"candidate_env",r->env,sizeof(r->env))||!field(line,"control_value",r->control,sizeof(r->control))||
           !field(line,"candidate_value",r->candidate,sizeof(r->candidate))||!allowed(r->env,r->control)||!allowed(r->env,r->candidate)||
           r->prefix<1||r->warmup<1||r->warmup>128||r->blocks<3||r->blocks>BLOCKS||r->vocab<2||r->layers<32||
           (r->depth!=2&&r->depth!=3)||integer(line,"measured")!=128||integer(line,"prefill_chunk")!=8192||
           (int64_t)r->ctx<(int64_t)r->prefix+132)r->errors++;
    } else if(!strncmp(line,"ordinary_selection ",19)) {
        if(r->selections>=FRONTIERS){r->errors++;return;}
        int at=integer(line,"offset"),a=integer(line,"ordinary_id"),b=integer(line,"mtp_trajectory_id");
        if(at!=r->selections||a<0||a>=r->vocab||b<0||b>=r->vocab||
           !is(line,"agrees",a==b?"yes":"no")||!is(line,"evidence_only","yes"))r->errors++;
        r->trajectory_ids[r->selections]=b;
        r->selections++;r->agreements+=a==b;
    } else if(!strncmp(line,"ordinary_drift ",15)) {
        double maximum,mean;unsigned long long changed;
        if(r->drifts>=FRONTIERS){r->errors++;return;}
        r->drift_offsets[r->drifts]=integer(line,"offset");
        if(integer(line,"public_frontier")!=r->drifts||integer(line,"count")!=r->vocab||
           !uint_field(line,"differing_logits",&changed)||changed>(unsigned)r->vocab||
           !number(line,"max_abs",&maximum)||!number(line,"mean_abs",&mean)||mean>maximum+1e-12||
           (!changed&&(maximum!=0||mean!=0))||!is(line,"interpretation","reported_not_thresholded"))r->errors++;
        r->drifts++;
    } else if(!strncmp(line,"ordinary_evidence ",18)) {
        r->declared_agreements=integer(line,"selected_agreements");r->declared_frontiers=integer(line,"public_frontiers");
        if(++r->evidence!=1||integer(line,"selected_comparisons")!=129||
           !is(line,"trajectory","teacher_forced_control_mtp")||!is(line,"quality","not_decided_by_this_tool")||
           !is(line,"full_state_equivalence","not_claimed"))r->errors++;
    } else if(!strncmp(line,"reference ",10)) {
        r->reference_cycles=integer(line,"measured_cycles");r->warm_cycles=integer(line,"warmup_cycles");
        r->continuation_cycles=integer(line,"restored_continuation_cycles");
        if(++r->reference!=1||!mode(line)||r->reference_cycles<1||r->reference_cycles>128||
           integer(line,"public_frontiers")!=r->reference_cycles+1||r->warm_cycles<1||r->warm_cycles>r->warmup||
           r->continuation_cycles!=2||integer(line,"first_cycle_tokens")!=1||
           !is(line,"intermediate_snapshot_identity","full_payload_sha256")||!is(line,"final_snapshot_identity","raw_bytes")||
           !is(line,"ordinary_quality","pending"))r->errors++;
    } else if(!strncmp(line,"reference_frontier ",19)) {
        if(r->nrefs>=FRONTIERS){r->errors++;return;}
        frontier *f=&r->refs[r->nrefs];f->offset=integer(line,"offset");f->n=integer(line,"committed");f->id=integer(line,"selected");
        if(!mode(line)||integer(line,"index")!=r->nrefs||f->id<0||f->id>=r->vocab||
           !uint_field(line,"snapshot_bytes",&f->bytes)||!f->bytes||!digest(line,"snapshot_sha256",f->state)||
           !digest(line,"logits_sha256",f->logits)||
           (r->nrefs?(f->n<1||f->n>r->depth||(int64_t)f->offset!=(int64_t)r->refs[r->nrefs-1].offset+f->n):(f->offset!=0||f->n!=0))||
           (r->nrefs==1&&f->n!=1)||f->offset>128)r->errors++;
        r->nrefs++;
    } else if(!strncmp(line,"reference_hashes ",17)) {
        char expected[65],trajectory[65];unsigned char tokens[129*4];plan_hash(r,expected);
        for(int i=0;i<129;i++)for(unsigned j=0;j<4;j++)tokens[i*4+j]=(unsigned char)((uint32_t)r->trajectory_ids[i]>>(j*8));
        sha_bytes(tokens,sizeof(tokens),trajectory);
        if(++r->refhash!=1||integer(line,"trajectory_tokens")!=129||integer(line,"cycle_plan_frontiers")!=r->nrefs||
           !is(line,"integer_encoding","u32_little_endian")||!digest(line,"trajectory_sha256",r->trajectory)||
           !digest(line,"cycle_plan_sha256",r->plan)||strcmp(expected,r->plan)||strcmp(trajectory,r->trajectory))r->errors++;
    } else if(!strncmp(line,"validation ",11)) {
        int v=variant(line),replay=integer(line,"replay");
        if(v<0||replay<1||replay>2){r->errors++;return;}
        const int bit=1<<(replay-1);if(r->validation[v]&bit)r->errors++;r->validation[v]|=bit;
        if(!mode(line)||!is(line,"all_public_frontier_logits_exact","yes")||
           !is(line,"all_public_frontier_full_snapshot_sha256_exact","yes")||!is(line,"final_snapshot_bytes_exact","yes")||
           !is(line,"cycle_token_selection_exact","yes")||!is(line,"restored_continuation_exact","yes"))r->errors++;
    } else if(!strncmp(line,"run=",4)) {
        if(r->runs>=RUNS){r->errors++;return;}
        const int index=r->runs,block=index/4,slot=index%4,expected=(block&1)?(slot==0||slot==3):(slot==1||slot==2);
        sample *s=&r->samples[index];s->variant=variant(line);s->cycles=integer(line,"cycles");
        double tps=0,mean=0;
        if((index&&!r->samples[index-1].memory)||r->validation[0]!=3||r->validation[1]!=3||
           !mode(line)||integer(line,"run")!=index+1||integer(line,"repeat")!=block+1||integer(line,"slot")!=slot+1||
           !is(line,"pattern",block&1?"BAAB":"ABBA")||s->variant!=expected||integer(line,"tokens")!=128||
           s->cycles!=r->reference_cycles||!number(line,"seconds",&s->seconds)||s->seconds<=0||
           !number(line,"tokens_per_second",&tps)||!rate_matches(s->seconds,tps,128)||
           !number(line,"first_cycle_ms",&s->first_ms)||!number(line,"mean_token_ms",&mean)||
           !near(mean,1000*s->seconds/128,0.00000052)||!pending(line)||
           !is(line,"cycle_token_selection_exact","yes")||!is(line,"final_logits_exact","yes")||
           !is(line,"final_snapshot_bytes_exact","yes")||!is(line,"restored_continuation_exact","yes")||
           !is(line,"reference","control_mtp_same_depth"))r->errors++;
        if(s->variant>=0&&s->variant<=1){r->seconds[s->variant]+=s->seconds;r->first_ms[s->variant]+=s->first_ms;r->n[s->variant]++;}
        r->runs++;
    } else if(!strncmp(line,"frontier ",9)) {
        int run=integer(line,"run");if(run<1||run!=r->runs||!r->nrefs){r->errors++;return;}
        sample *s=&r->samples[run-1];frontier *f=&r->refs[r->nrefs-1];char a[65],b[65],c[65],d[65];unsigned long long bytes;
        if(++s->hash!=1||integer(line,"tokens")!=(int64_t)r->prefix+128||!uint_field(line,"checkpoint_bytes",&bytes)||bytes!=f->bytes||
           !digest(line,"checkpoint_sha256",a)||strcmp(a,f->state)||!digest(line,"logits_sha256",b)||strcmp(b,f->logits)||
           !digest(line,"trajectory_sha256",c)||strcmp(c,r->trajectory)||!digest(line,"cycle_plan_sha256",d)||strcmp(d,r->plan)||
           !is(line,"scope","timed_final_and_checked_cycle_trajectory"))r->errors++;
    } else if(!strncmp(line,"cycle ",6)) {
        int run=integer(line,"run");if(run<1||run!=r->runs){r->errors++;return;}
        sample *s=&r->samples[run-1];const int c=s->cycle_rows;double seconds;
        if(c>=r->reference_cycles||c+1>=r->nrefs){r->errors++;return;}
        if(s->resources||!s->hash||r->refs[c+1].n<1||r->refs[c+1].n>3||integer(line,"cycle")!=c+1||integer(line,"offset")!=r->refs[c].offset||
           integer(line,"tokens")!=r->refs[c+1].n||!number(line,"seconds",&seconds)||seconds<=0)r->errors++;
        else {s->cycle_sum+=seconds;s->offset+=r->refs[c+1].n;if(!c&&!near(s->first_ms,1000*seconds,0.0000016))r->errors++;}
        s->cycle_rows++;
    } else if(!strncmp(line,"resources ",10)) {
        int run=integer(line,"run");if(run<1||run!=r->runs){r->errors++;return;}
        sample *s=&r->samples[run-1];unsigned long long value;double user,system;
        const char *keys[]={"maxrss","minor_faults","major_faults","input_blocks","output_blocks","swaps"};
        if(++s->resources!=1||s->cycle_rows!=s->cycles)r->errors++;
        for(unsigned i=0;i<sizeof(keys)/sizeof(keys[0]);i++)if(!uint_field(line,keys[i],&value))r->errors++;
        if(!number(line,"user_seconds",&user)||!number(line,"system_seconds",&system))r->errors++;
    } else if(!strncmp(line,"memory ",7)) {
        int run=integer(line,"run");if(run<1||run!=r->runs){r->errors++;return;}
        sample *s=&r->samples[run-1];s->process_valid=integer(line,"process_valid");s->swap_valid=integer(line,"swap_valid");s->vm_valid=integer(line,"vm_valid");
        if(++s->memory!=1||s->resources!=1||s->process_valid<0||s->process_valid>1||s->swap_valid<0||s->swap_valid>1||s->vm_valid<0||s->vm_valid>1||
           !uint_field(line,"resident_bytes",&s->resident)||!uint_field(line,"footprint_bytes",&s->footprint)||
           !uint_field(line,"disk_read_bytes",&s->reads)||!uint_field(line,"disk_write_bytes",&s->writes)||
           !uint_field(line,"global_swap_used_bytes",&s->swap)||!uint_field(line,"global_swapins_pages",&s->swapins)||
           !uint_field(line,"global_swapouts_pages",&s->swapouts)||!is(line,"frontier","after_recovery_and_pool_drain"))r->errors++;
    } else if(!strncmp(line,"paired ",7)) {
        if(r->paired>=BLOCKS){r->errors++;return;}
        if(!mode(line)||integer(line,"repeat")!=r->paired+1||r->runs!=(r->paired+1)*4||
           !finite_number(line,"gain_percent",&r->paired_printed[r->paired]))r->errors++;
        r->paired++;
    } else if(!strncmp(line,"paired_ci ",10)) {
        if(++r->summary!=1||!mode(line)||integer(line,"blocks")!=r->blocks||!pending(line)||
           !is(line,"same_depth_equivalence","yes")||!is(line,"ordinary_equivalence","not_claimed")||!is(line,"accepted","no")||
           !finite_number(line,"gain_percent",&r->summary_values[0])||!finite_number(line,"low95_percent",&r->summary_values[1])||
           !finite_number(line,"high95_percent",&r->summary_values[2]))r->errors++;
    } else if(!strncmp(line,"aggregate ",10)) {
        int v=variant(line);if(v<0){r->errors++;return;}double seconds,tps;
        if(++r->aggregate[v]!=1||!mode(line)||integer(line,"runs")!=r->n[v]||integer(line,"tokens")!=128*r->n[v]||
           !number(line,"seconds",&seconds)||!number(line,"tokens_per_second",&tps)||
           !near(seconds,r->seconds[v],(r->n[v]+1)*0.00000000051)||!rate_matches(seconds,tps,(unsigned)r->n[v]*128))r->errors++;
    } else if(!strncmp(line,"single_summary ",15)) {r->single=1;r->errors++;}
}

static void finish(report *r) {
    r->gain=r->low=r->high=NAN;
    if(r->config!=1||r->single||r->reference!=1||r->refhash!=1||r->nrefs!=r->reference_cycles+1||!r->nrefs||
       r->refs[r->nrefs-1].offset!=128||r->selections!=129||r->evidence!=1||r->declared_agreements!=r->agreements||
       r->declared_frontiers!=r->nrefs||r->drifts!=r->nrefs||r->validation[0]!=3||r->validation[1]!=3||
       r->blocks<3||r->blocks>BLOCKS||r->runs!=r->blocks*4||r->paired!=r->blocks||r->summary!=1||
       r->aggregate[0]!=1||r->aggregate[1]!=1||r->n[0]!=2*r->blocks||r->n[1]!=2*r->blocks)r->errors++;
    for(int i=0;i<r->nrefs;i++)if(r->drift_offsets[i]!=r->refs[i].offset||r->refs[i].offset<0||r->refs[i].offset>128||r->refs[i].id!=r->trajectory_ids[r->refs[i].offset])r->errors++;
    for(int i=0;i<r->runs;i++){sample *s=&r->samples[i];if(s->cycle_rows!=s->cycles||s->offset!=128||s->resources!=1||s->memory!=1||s->hash!=1||s->cycle_sum>s->seconds+s->cycles*0.000000001)r->errors++;}
    if(r->errors)return;
    double ratios[BLOCKS],sum=0,variance=0,minimum=INFINITY;
    for(int b=0;b<r->blocks;b++) {
        double seconds[2]={0};for(int slot=0;slot<4;slot++){sample *s=&r->samples[b*4+slot];seconds[s->variant]+=s->seconds;if(s->seconds<minimum)minimum=s->seconds;}
        ratios[b]=log(seconds[0]/seconds[1]);sum+=ratios[b];
        if(!near(100*expm1(ratios[b]),r->paired_printed[b],0.000001+300e-9/minimum))r->errors++;
    }
    const double mean=sum/r->blocks;
    for(int b=0;b<r->blocks;b++){double d=ratios[b]-mean;variance+=d*d;}
    const int df=r->blocks-1;const double t=critical[df<30?df:30];
    const double radius=t*sqrt(variance/df/r->blocks);
    r->gain=100*expm1(mean);r->low=100*expm1(mean-radius);r->high=100*expm1(mean+radius);
    const double tolerance=0.00002+1600*t*1e-9/minimum;
    if(!isfinite(r->gain)||!isfinite(r->low)||!isfinite(r->high)||
       !near(r->gain,r->summary_values[0],tolerance)||!near(r->low,r->summary_values[1],tolerance)||!near(r->high,r->summary_values[2],tolerance))r->errors++;
    r->valid=!r->errors;if(!r->valid)r->gain=r->low=r->high=NAN;
}

typedef struct { char hash[65],path[PATHCAP]; } hash_entry;
typedef struct {hash_entry *entries;int count;} manifest;
static const char *base_name(const char *p){const char *slash=strrchr(p,'/');return slash?slash+1:p;}
static int joined(char out[PATHCAP],const char *dir,const char *name) {return snprintf(out,PATHCAP,"%s/%s",dir,name)<PATHCAP;}
static int read_manifest(const char *path,manifest *m) {
    FILE *f=fopen(path,"r");if(!f)return 0;char *line=NULL;size_t size=0;int okay=1;
    while(getline(&line,&size,f)>=0) {
        const size_t n=strlen(line);if(n<68||line[64]!=' '||line[65]!=' '||m->count>=4096){okay=0;break;}
        line[strcspn(line,"\r\n")]=0;line[64]=0;
        if(!valid_hash(line)||!line[66]||strlen(line+66)>=PATHCAP){okay=0;break;}
        for(int i=0;i<m->count;i++)if(!strcmp(m->entries[i].path,line+66)){okay=0;break;}
        if(!okay)break;
        hash_entry *next=realloc(m->entries,((size_t)m->count+1)*sizeof(*next));if(!next){okay=0;break;}m->entries=next;
        strcpy(next[m->count].hash,line);strcpy(next[m->count].path,line+66);m->count++;
    }
    if(ferror(f))okay=0;free(line);fclose(f);return okay&&m->count>0;
}
static const char *hash_for(const manifest *m,const char *path) {for(int i=0;i<m->count;i++)if(!strcmp(m->entries[i].path,path))return m->entries[i].hash;return NULL;}
static int recorded_in(const manifest *m,const char *path,const char *hash) {const char *p=hash_for(m,path);return p&&!strcmp(p,hash);}
static char *read_file(const char *path,size_t *bytes) {
    FILE *f=fopen(path,"rb");if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;}long n=ftell(f);
    if(n<0||n>32*1024*1024||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *p=calloc((size_t)n+1,1);if(!p){fclose(f);return NULL;}
    if(fread(p,1,(size_t)n,f)!=(size_t)n){free(p);fclose(f);return NULL;}
    fclose(f);if(bytes)*bytes=(size_t)n;return p;
}
typedef struct {manifest results,source,runtime,diagnostic;char repo[PATHCAP],bench[PATHCAP],model[PATHCAP],prompt[PATHCAP];int okay;} provenance;
static int protocol(provenance *p,const char *dir) {
    char path[PATHCAP];if(!joined(path,dir,"results.sha256")||!read_manifest(path,&p->results))return 0;
    /* Result manifests record absolute paths. Relocation uses unique basename
     * within the archived screen directory; source identities remain recorded. */
    for(int i=0;i<p->results.count;i++) {
        const char *name=base_name(p->results.entries[i].path);char hash[65];
        for(int j=0;j<i;j++)if(!strcmp(name,base_name(p->results.entries[j].path)))return 0;
        if(!joined(path,dir,name)||!sha_file(path,hash)||strcmp(hash,p->results.entries[i].hash))return 0;
    }
    const char *required[]={"source.sha256","runtime.sha256","diagnostic.sha256","build-settings.txt","revision.txt","model-stat.txt","protocol.txt","exit-codes.tsv"};
    for(unsigned i=0;i<sizeof(required)/sizeof(required[0]);i++) {
        int found=0;for(int j=0;j<p->results.count;j++)found+=!strcmp(required[i],base_name(p->results.entries[j].path));
        if(found!=1)return 0;
    }
    if(!joined(path,dir,"source.sha256")||!read_manifest(path,&p->source)||
       !joined(path,dir,"runtime.sha256")||!read_manifest(path,&p->runtime)||
       !joined(path,dir,"diagnostic.sha256")||!read_manifest(path,&p->diagnostic)||!joined(path,dir,"protocol.txt"))return 0;
    char *text=read_file(path,NULL);if(!text)return 0;
    int okay=is(text,"format","mtp-same-depth-screen-v1")&&mode(text)&&
        field(text,"repo",p->repo,sizeof(p->repo))&&field(text,"bench",p->bench,sizeof(p->bench))&&
        field(text,"model",p->model,sizeof(p->model))&&field(text,"prompt",p->prompt,sizeof(p->prompt))&&
        integer(text,"prefix")==8192&&integer(text,"ctx")==8449&&integer(text,"chunk")==8192&&integer(text,"warmup")==16&&
        integer(text,"tokens")==128&&integer(text,"repeats")==3&&integer(text,"mtp_depth")==2&&
        is(text,"selection","non_eos_argmax")&&is(text,"environment","clean_HOME_PATH_TMPDIR_C_locale")&&
        integer(text,"telemetry_calls_per_run")==3&&is(text,"telemetry_call_evidence","wrapper_source_and_CPU_mock")&&
        is(text,"ordinary_equivalence","not_claimed")&&is(text,"quality","pending")&&is(text,"accepted","no");
    free(text);if(!okay)return 0;
    for(int i=0;i<p->runtime.count;i++) {
        if(p->runtime.entries[i].path[0]=='/'||strstr(p->runtime.entries[i].path,"..")||
           !joined(path,p->repo,p->runtime.entries[i].path)||!recorded_in(&p->source,path,p->runtime.entries[i].hash))return 0;
    }
    for(int i=0;i<p->diagnostic.count;i++)if(!recorded_in(&p->source,p->diagnostic.entries[i].path,p->diagnostic.entries[i].hash))return 0;
    const char *core[]={"ds4.c","ds4_metal.m","ds4.h","ds4_gpu.h","ds4.o","ds4_image.o","ds4_distributed.o","ds4_tp.o","ds4_ssd.o","ds4_metal.o","ds4_layer_pack.o","ds4_engram.o","speed-bench/benchmark_pool.h","speed-bench/benchmark_pool.m","speed-bench/benchmark_pool.o","speed-bench/metal_decode_schedule_bench.c","metal/qwen4.metal"};
    for(unsigned i=0;i<sizeof(core)/sizeof(core[0]);i++)if(!hash_for(&p->runtime,core[i]))return 0;
    if(!hash_for(&p->source,p->bench)||!hash_for(&p->source,p->prompt)||!hash_for(&p->diagnostic,p->bench))return 0;
    const char *wrapper=NULL,*strict=NULL;
    for(int i=0;i<p->diagnostic.count;i++) {
        const char *name=base_name(p->diagnostic.entries[i].path);
        if(!strcmp(name,"metal_decode_mtp_same_depth.c"))wrapper=p->diagnostic.entries[i].hash;
        if(!strcmp(name,"metal_decode_schedule_bench.c"))strict=p->diagnostic.entries[i].hash;
    }
    return wrapper&&strict&&!strcmp(strict,hash_for(&p->runtime,"speed-bench/metal_decode_schedule_bench.c"));
}
static int result_recorded(const provenance *p,const char *basename) {
    int n=0;for(int i=0;i<p->results.count;i++)n+=!strcmp(base_name(p->results.entries[i].path),basename);return n==1;
}
static int sidecars(const provenance *p,const char *dir,const report *r) {
    char path[PATHCAP],name[PATHCAP];const char *base=base_name(r->path);const char *suffix[]={"",".exit",".argv",".arguments.txt"};
    for(unsigned i=0;i<4;i++){if(snprintf(name,sizeof(name),"%s%s",base,suffix[i])>=(int)sizeof(name)||!result_recorded(p,name))return 0;}
    char actual[65];if(!sha_file(r->path,actual))return 0;
    for(int i=0;i<p->results.count;i++)if(!strcmp(base,base_name(p->results.entries[i].path))&&strcmp(actual,p->results.entries[i].hash))return 0;
    snprintf(name,sizeof(name),"%s.exit",base);if(!joined(path,dir,name))return 0;
    char *exit_text=read_file(path,NULL);int okay=exit_text&&!strcmp(exit_text,"0\n");free(exit_text);if(!okay)return 0;
    snprintf(name,sizeof(name),"%s.argv",base);if(!joined(path,dir,name))return 0;
    size_t bytes=0;char *argv=read_file(path,&bytes);if(!argv||!bytes||argv[bytes-1]){free(argv);return 0;}
    const char *expected[]={p->bench,"--model",r->model,"--prompt-file",r->prompt,"--prefix-tokens","8192","--ctx","8449", "--prefill-chunk","8192","--warmup","16","--tokens","128","--repeats","3","--include-selection","--mtp","--mtp-depth","2","--candidate-env",r->env,"--control-value",r->control,"--candidate-value",r->candidate};
    size_t at=0;for(unsigned i=0;i<sizeof(expected)/sizeof(expected[0]);i++){size_t n=strlen(expected[i])+1;if(n>bytes-at||memcmp(argv+at,expected[i],n)){okay=0;break;}at+=n;}
    if(at!=bytes)okay=0;
    for(size_t i=0;i<bytes;i++)if(!argv[i])argv[i]='\n';
    snprintf(name,sizeof(name),"%s.arguments.txt",base);if(!joined(path,dir,name)){free(argv);return 0;}
    size_t arguments_bytes=0;char *arguments=read_file(path,&arguments_bytes);
    if(!arguments||arguments_bytes!=bytes||memcmp(arguments,argv,bytes))okay=0;free(arguments);free(argv);
    if(strcmp(r->model,p->model)||strcmp(r->prompt,p->prompt)||r->prefix!=8192||r->ctx!=8449||r->warmup!=16||r->blocks!=3||r->depth!=2)okay=0;
    if(!joined(path,dir,"exit-codes.tsv"))return 0;char *table=read_file(path,NULL);if(!table)return 0;
    int matches=0;char *save=NULL;
    for(char *line=strtok_r(table,"\n",&save);line;line=strtok_r(NULL,"\n",&save)) {
        char *cols[8];int n=0;char *cursor=line;
        while(n<8){cols[n++]=cursor;char *tab=strchr(cursor,'\t');if(!tab)break;*tab=0;cursor=tab+1;}
        if(n==8&&!strcmp(base_name(cols[7]),base)){
            matches++;if(strcmp(cols[1],"mtp_same_depth")||strcmp(cols[2],r->env)||strcmp(cols[3],r->control)||strcmp(cols[4],r->candidate)||strcmp(cols[5],"0")||strcmp(cols[6],"completed_requires_analysis"))okay=0;
        }
    }
    free(table);return okay&&matches==1;
}
static void quoted(FILE *f,const char *s){fputc('"',f);for(;*s;s++){if(*s=='"')fputc('"',f);fputc(*s,f);}fputc('"',f);}
static void xml(FILE *f,const char *s){for(;*s;s++)switch(*s){case '&':fputs("&amp;",f);break;case '<':fputs("&lt;",f);break;case '>':fputs("&gt;",f);break;case '"':fputs("&quot;",f);break;default:fputc(*s,f);}}
static void metrics(FILE *csv,const report *r,int v) {
    if(!r->valid){for(int i=0;i<16;i++)fputc(',',csv);return;}
    int process=0,swap_valid=0,vm=0,cycles=0;
    unsigned long long resident=0,footprint=0,swap=0;long double reads=0,writes=0,swapins=0,swapouts=0;
    for(int i=0;i<r->runs;i++) {
        const sample *s=&r->samples[i];if(s->variant!=v)continue;cycles+=s->cycles;
        if(s->process_valid){process++;if(s->resident>resident)resident=s->resident;if(s->footprint>footprint)footprint=s->footprint;reads+=s->reads;writes+=s->writes;}
        if(s->swap_valid){swap_valid++;if(s->swap>swap)swap=s->swap;}
        if(s->vm_valid){vm++;swapins+=s->swapins;swapouts+=s->swapouts;}
    }
    const double n=r->n[v];
    fprintf(csv,",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,",1000*r->seconds[v]/n,1000*r->seconds[v]/(128*n),r->first_ms[v]/n,cycles/n,128*n/cycles,1000*r->seconds[v]/cycles,process);
    if(process)fprintf(csv,"%llu,%llu,%.0Lf,%.0Lf",resident,footprint,reads,writes);else fputs(",,,",csv);
    fprintf(csv,",%d,",swap_valid);if(swap_valid)fprintf(csv,"%llu",swap);
    fprintf(csv,",%d,",vm);if(vm)fprintf(csv,"%.0Lf,%.0Lf",swapins,swapouts);else fputc(',',csv);
}
static void output(FILE *csv,FILE *svg,report *r,int count,int provenance_okay) {
    fputs("log,mode,status,accepted,quality,independent_recovery_and_component_gates,provenance,depth,blocks,runs,public_frontiers,ordinary_selected_agreements,ordinary_selected_comparisons,control_tokens_per_second,candidate_tokens_per_second,paired_gain_percent,low95_percent,high95_percent,untimed_public_logits_exact,untimed_public_full_snapshot_identity,timed_final_logits_and_snapshot_bytes_exact,timed_cycle_token_selection_exact,ordinary_equivalence,resource_rows,telemetry_call_count_basis,resource_frontier,candidate_env,control_value,candidate_value,configuration",csv);
    const char *names[]={"mean_timed_region_ms","mean_committed_token_ms","mean_first_mtp_cycle_ms","mean_cycles_per_run","mean_tokens_per_cycle","mean_mtp_cycle_ms","valid_process_rows","max_resident_bytes","max_footprint_bytes","timed_disk_read_bytes","timed_disk_write_bytes","valid_swap_rows","max_global_swap_used_bytes","valid_vm_rows","timed_global_swapins_pages","timed_global_swapouts_pages"};
    for(int v=0;v<2;v++)for(unsigned i=0;i<sizeof(names)/sizeof(names[0]);i++)fprintf(csv,",%s_%s",v?"candidate":"control",names[i]);
    fputc('\n',csv);
    double lo=-5,hi=5;for(int i=0;i<count;i++)if(r[i].valid){lo=fmin(lo,r[i].low);hi=fmax(hi,r[i].high);}lo=floor(lo/5)*5;hi=ceil(hi/5)*5;
    fprintf(svg,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"%d\"><title>Same-depth MTP screens; quality and acceptance pending</title><rect width=\"100%%\" height=\"100%%\" fill=\"white\"/><g fill=\"#173346\" font-family=\"Arial,sans-serif\"><text x=\"20\" y=\"30\" font-size=\"20\">Same-depth MTP paired screens — acceptance pending</text><text x=\"20\" y=\"53\" font-size=\"12\">Exact same-depth replay; ordinary teacher-forced drift is separate evidence. No quality decision.</text>",100+count*55);
    for(int i=0;i<count;i++) {
        const char *status=r[i].single?"unsupported_single_run":!r[i].valid?"invalid_screen":r[i].low>0?"positive_paired_ci":r[i].high<0?"negative_paired_ci":"inconclusive";
        quoted(csv,r[i].path);fprintf(csv,",mtp_same_depth,%s,no,pending,external_required,%s,%d,%d,%d,%d,%d,129,",status,provenance_okay?"recorded_manifest_hashes_valid":"invalid",r[i].depth,r[i].blocks,r[i].runs,r[i].nrefs,r[i].agreements);
        if(r[i].valid)fprintf(csv,"%.9g,%.9g,%.9g,%.9g,%.9g,yes,full_payload_sha256,yes,yes",128.0*r[i].n[0]/r[i].seconds[0],128.0*r[i].n[1]/r[i].seconds[1],r[i].gain,r[i].low,r[i].high);
        else fputs(",,,,,not_established,not_established,not_established,not_established",csv);
        fprintf(csv,",not_claimed,%d,three_calls_in_pinned_source_and_CPU_fixture,after_recovery_and_pool_drain,",r[i].valid?r[i].runs:0);
        quoted(csv,r[i].env);fputc(',',csv);quoted(csv,r[i].control);fputc(',',csv);quoted(csv,r[i].candidate);fputc(',',csv);quoted(csv,r[i].config_line);metrics(csv,&r[i],0);metrics(csv,&r[i],1);fputc('\n',csv);
        const int y=88+i*55;fprintf(svg,"<text x=\"20\" y=\"%d\" font-size=\"12\">",y);xml(svg,base_name(r[i].path));fputs("</text>",svg);
        if(r[i].valid){fprintf(svg,"<path d=\"M%.2f %d H%.2f\" stroke=\"#9b711e\" stroke-width=\"2\"/><circle cx=\"%.2f\" cy=\"%d\" r=\"4\" fill=\"#9b711e\"/><text x=\"620\" y=\"%d\" font-size=\"12\">%+.3f%% [%+.3f, %+.3f]</text>",620+420*(r[i].low-lo)/(hi-lo),y-5,620+420*(r[i].high-lo)/(hi-lo),620+420*(r[i].gain-lo)/(hi-lo),y-5,y+17,r[i].gain,r[i].low,r[i].high);}
        else fprintf(svg,"<text x=\"620\" y=\"%d\" font-size=\"12\" fill=\"#a33\">Invalid or unsupported evidence</text>",y);
    }
    fputs("</g></svg>\n",svg);
}
static int same_file(const char *a,const char *b){if(!strcmp(a,b))return 1;struct stat x,y;return !stat(a,&x)&&!stat(b,&y)&&x.st_dev==y.st_dev&&x.st_ino==y.st_ino;}
#ifndef SD_ANALYZER_ENTRY
#define SD_ANALYZER_ENTRY main
#endif
int SD_ANALYZER_ENTRY(int argc,char **argv) {
    if(argc<5||argc>FILES+4){fprintf(stderr,"usage: %s NEW.csv NEW.svg SCREEN_DIRECTORY PAIRED_LOG...\n",argv[0]);return 2;}
    if(same_file(argv[1],argv[2]))return 2;
    for(int i=4;i<argc;i++)if(same_file(argv[1],argv[i])||same_file(argv[2],argv[i]))return 2;
    provenance p={0};p.okay=protocol(&p,argv[3]);if(!p.okay)fprintf(stderr,"Invalid or incomplete recorded screen provenance: %s\n",argv[3]);
    const int count=argc-4;report *r=calloc((size_t)count,sizeof(*r));if(!r)return 2;int bad=0;
    for(int i=0;i<count;i++) {
        r[i].path=argv[i+4];FILE *f=fopen(r[i].path,"r");
        if(!f)r[i].errors++;else{char *line=NULL;size_t cap=0;while(getline(&line,&cap,f)>=0){consume(&r[i],line);if(r[i].errors>1024)break;}if(ferror(f))r[i].errors++;free(line);fclose(f);}
        if(!p.okay||!sidecars(&p,argv[3],&r[i]))r[i].errors++;
        finish(&r[i]);if(!r[i].valid){fprintf(stderr,"Invalid paired same-depth log: %s (%d errors)\n",r[i].path,r[i].errors);bad=1;}
    }
    int cfd=open(argv[1],O_WRONLY|O_CREAT|O_EXCL,0644),sfd=-1;
    if(cfd>=0)sfd=open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644);
    if(cfd<0||sfd<0){if(cfd>=0){close(cfd);unlink(argv[1]);}free(r);free(p.results.entries);free(p.source.entries);free(p.runtime.entries);free(p.diagnostic.entries);return 2;}
    FILE *csv=fdopen(cfd,"w"),*svg=fdopen(sfd,"w");
    if(!csv||!svg){if(csv)fclose(csv);else close(cfd);if(svg)fclose(svg);else close(sfd);free(r);free(p.results.entries);free(p.source.entries);free(p.runtime.entries);free(p.diagnostic.entries);return 2;}
    output(csv,svg,r,count,p.okay);int failed=ferror(csv)||ferror(svg);if(fclose(csv))failed=1;if(fclose(svg))failed=1;
    free(r);free(p.results.entries);free(p.source.entries);free(p.runtime.entries);free(p.diagnostic.entries);return failed?2:bad?1:0;
}
