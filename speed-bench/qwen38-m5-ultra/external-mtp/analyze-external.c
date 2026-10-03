/* External fixed-depth2 MTP ABBA. The copied paired analyzer supplies lexical,
 * hash and shared frontier parsing only; its paired entry remains separate.
 * Copy/hash the reviewed parser beside this file before compilation. */
#define SD_ANALYZER_ENTRY ds4_unused_paired_analyzer_entry
#include "analyze-same-depth.c"
#undef SD_ANALYZER_ENTRY
#include <limits.h>
enum { EXT_CASES=8, EXT_RUNS=128 };
typedef struct {
    report r;int present,variant,provenance,terminal,selected[129],selection_count;
    char runtime[65],binary[65],args[16384],settings[512];
} external_sample;
typedef struct {
    char name[128];external_sample *runs[EXT_RUNS];int count,valid,blocks;
    double gain,low,high,tps[2];
} external_case;
static int ext_literal(const char *line,const char *literal){
    const char *colon=strchr(literal,':');char key[128];if(!colon||(size_t)(colon-literal)+2>sizeof(key))return 0;
    size_t n=(size_t)(colon-literal)+1;memcpy(key,literal,n);key[n]=0;const char *p=strstr(line,key);
    return p&&!strstr(p+n,key)&&!strncmp(p,literal,strlen(literal));
}
static int ext_hash(const char *line,const char *key,char out[65]){
    char find[160];snprintf(find,sizeof(find),"\"%s\":\"",key);const char *p=strstr(line,find);
    if(!p)return 0;p+=strlen(find);if(strstr(p,find)||strlen(p)<65||p[64]!='"')return 0;
    memcpy(out,p,64);out[64]=0;return valid_hash(out);
}
static int ext_expected(const char *line,const char *key,const char *value){char hash[65];return ext_hash(line,key,hash)&&!strcmp(hash,value);}
static void ext_json(FILE *f,const char *value){
    fputc('"',f);for(const unsigned char *p=(const unsigned char*)value;*p;p++){
        if(*p=='"'||*p=='\\'){fputc('\\',f);fputc(*p,f);}else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);
    }fputc('"',f);
}
static char *ext_expected_args(const report *r){
    char prefix[32],ctx[32];snprintf(prefix,sizeof(prefix),"%d",r->prefix);snprintf(ctx,sizeof(ctx),"%d",r->ctx);
    const char *args[]={"--single-run","--model",r->model,"--prompt-file",r->prompt,"--prefix-tokens",prefix,"--ctx",ctx,
        "--prefill-chunk","8192","--warmup","16","--tokens","128","--include-selection","--mtp","--mtp-depth","2"};
    char *text=NULL;size_t size=0;FILE *f=open_memstream(&text,&size);if(!f)return NULL;
    fputs("\"command_args\":[",f);for(unsigned i=0;i<sizeof(args)/sizeof(args[0]);i++){if(i)fputc(',',f);ext_json(f,args[i]);}fputs("]}",f);
    if(fclose(f)){free(text);return NULL;}return text;
}
static int ext_sidecars(external_sample *e){
    const char *log=e->r.path;char path[PATHCAP];manifest hashes={0};
    if(snprintf(path,sizeof(path),"%s.sha256",log)>=(int)sizeof(path)||!read_manifest(path,&hashes))return 0;
    const char *suffix[]={"",".argv",".arguments.txt",".exit"};int okay=hashes.count==4;
    for(unsigned i=0;i<4;i++){
        if(snprintf(path,sizeof(path),"%s%s",log,suffix[i])>=(int)sizeof(path)){okay=0;break;}
        char hash[65];int matches=0;
        for(int j=0;j<hashes.count;j++){
            if(!strcmp(base_name(path),base_name(hashes.entries[j].path))){
                matches++;if(!sha_file(path,hash)||strcmp(hash,hashes.entries[j].hash))okay=0;
            }
        }
        if(matches!=1)okay=0;
    }
    free(hashes.entries);if(!okay)return 0;
    snprintf(path,sizeof(path),"%s.exit",log);char *text=read_file(path,NULL);okay=text&&!strcmp(text,"0\n");free(text);if(!okay)return 0;
    snprintf(path,sizeof(path),"%s.argv",log);size_t size=0;char *argv=read_file(path,&size);if(!argv||!size||argv[size-1]){free(argv);return 0;}
    const char *words[64];int count=0;size_t pos=0;
    while(pos<size&&count<64){words[count++]=argv+pos;pos+=strlen(argv+pos)+1;}
    if(pos!=size||count<29){free(argv);return 0;}
    okay=!strcmp(words[0],"env")&&!strcmp(words[1],"-i")&&!strcmp(words[2],"PATH=/usr/bin:/bin:/usr/sbin:/sbin")&&
        !strncmp(words[3],"HOME=",5)&&!strncmp(words[4],"TMPDIR=",7)&&!strcmp(words[5],"LANG=C")&&!strcmp(words[6],"LC_ALL=C");
    int at=7;const char *names[5];int settings=0;
    while(at<count&&!strncmp(words[at],"DS4_",4)){
        const char *assignment=words[at],*equal=strchr(assignment,'=');
        const char *allowed_settings[]={"DS4_QWEN4_DECODE_FUSIONS=1","DS4_QWEN4_ATTN_QUERY_GROUP=4","DS4_QWEN4_ATTN_QUERY_GROUP=6","DS4_QWEN4_NGRAM_OVERLAP=1","DS4_QWEN4_GDN_NSG=4"};
        int known=0;for(unsigned i=0;i<5;i++)known|=!strcmp(assignment,allowed_settings[i]);
        if(!e->variant||!known||!equal||settings>=5){okay=0;break;}
        for(int i=0;i<settings;i++)if(!strncmp(names[i],assignment,(size_t)(equal-assignment)+1))okay=0;
        names[settings++]=assignment;at++;
    }
    for(int i=0;i<settings;i++)for(int j=i+1;j<settings;j++)if(strcmp(names[i],names[j])>0){const char *swap=names[i];names[i]=names[j];names[j]=swap;}
    size_t used=0;for(int i=0;i<settings;i++){int n=snprintf(e->settings+used,sizeof(e->settings)-used,"%s\n",names[i]);if(n<0||(size_t)n>=sizeof(e->settings)-used){okay=0;break;}used+=(size_t)n;}
    if(at>=count||strcmp(base_name(words[at++]),"run-mtp"))okay=0;
    if(e->variant){if(at>=count||strcmp(words[at++],"--candidate"))okay=0;}
    if(at>=count||strcmp(words[at++],"mtp"))okay=0;
    if(count-at<2||strcmp(words[count-2],"--output")||strcmp(words[count-1],log))okay=0;
    char *args=NULL;size_t args_size=0;FILE *f=open_memstream(&args,&args_size);if(!f){free(argv);return 0;}
    fputs("\"command_args\":[",f);for(int i=at;i<count-2;i++){if(i>at)fputc(',',f);ext_json(f,words[i]);}fputs("]}",f);if(fclose(f))okay=0;
    if(!args||strcmp(args,e->args))okay=0;free(args);
    for(size_t i=0;i<size;i++)if(!argv[i])argv[i]='\n';
    snprintf(path,sizeof(path),"%s.arguments.txt",log);size_t text_size=0;text=read_file(path,&text_size);
    if(!text||text_size!=size||memcmp(text,argv,size))okay=0;free(text);free(argv);return okay;
}
static void ext_consume(external_sample *e,const char *line){
    report *r=&e->r;
    sample *timed=&r->samples[0];
    /* Every mandatory record follows the producer's actual execution order. */
    if((!strncmp(line,"ordinary_selection ",19)&&(r->config!=1||r->evidence||r->reference||r->runs))||
       (!strncmp(line,"ordinary_drift ",15)&&(r->config!=1||r->evidence||r->reference||integer(line,"offset")!=r->selections-1))||
       (!strncmp(line,"ordinary_evidence ",18)&&(r->selections!=129||r->reference||r->runs))||
       (!strncmp(line,"reference ",10)&&(r->evidence!=1||r->selections!=129||r->nrefs||r->runs))||
       (!strncmp(line,"reference_frontier ",19)&&(r->reference!=1||r->refhash||r->validation[0]||r->runs))||
       (!strncmp(line,"reference_hashes ",17)&&(r->reference!=1||r->nrefs!=r->reference_cycles+1||r->validation[0]||r->runs))||
       (!strncmp(line,"validation ",11)&&(r->refhash!=1||r->nrefs!=r->reference_cycles+1||r->runs))||
       (!strncmp(line,"frontier ",9)&&(r->runs!=1||timed->cycle_rows||timed->resources||timed->memory||e->terminal))||
       (!strncmp(line,"cycle ",6)&&(r->runs!=1||timed->resources||timed->memory||e->terminal))||
       (!strncmp(line,"resources ",10)&&(r->runs!=1||timed->memory||e->terminal))||
       (!strncmp(line,"memory ",7)&&(r->runs!=1||timed->resources!=1||e->terminal))||
       (!strncmp(line,"aggregate ",10)&&(e->terminal!=1||timed->memory!=1))){r->errors++;return;}
    if(!strncmp(line,"build_provenance=",17)){
        char literal[64];snprintf(literal,sizeof(literal),"\"runtime\":\"%s\"",e->variant?"candidate":"original");
        const char *args=strstr(line,"\"command_args\":[");
        if(++e->provenance!=1||r->config||!ext_literal(line,literal)||!ext_literal(line,"\"format\":\"external-mtp-v1\"")||
           !ext_literal(line,"\"baseline_kind\":\"original\"")||!ext_literal(line,"\"verification_only\":false")||
           !ext_literal(line,"\"original_revision\":\"0aaea5a238fb41a35106a551e73c8409dfb751ac\"")||
           !ext_literal(line,"\"runtime_files\":167,")||!ext_literal(line,"\"metal_sources_pinned\":26,")||
           !ext_hash(line,"runtime_files_sha256",e->runtime)||!ext_hash(line,"binary_sha256",e->binary)||
           (!e->variant&&strcmp(e->runtime,"822fe7e1d0491f5b4a830147e79d2724667a576ce9f430bfc2da02eb392bd17d"))||
           !ext_expected(line,"wrapper_sha256","1555a405f6bd197eb961ba4bef7aee9784132fec96eae802e013851e2281b75c")||
           !ext_expected(line,"strict_protocol_sha256","8637e88c9dd7ed785a1e3ebcf34b8b3c6d03cee402295d801c34e64de2095f73")||
           !ext_expected(line,"pool_h_sha256","68dc3bcaf63252cee993679660b0ed7500db76b0fce71182f225a96365adc0be")||
           !ext_expected(line,"pool_m_sha256","dd19fbc06a02b406f8786220481994cb5ea2b6ff666d375ca5f232b90a5c16f9")||
           !args||strlen(args)>=sizeof(e->args))r->errors++;
        else {strcpy(e->args,args);e->args[strcspn(e->args,"\r\n")]=0;}
    }else if(!strncmp(line,"configuration ",14)){
        if(++r->config!=1||e->provenance!=1||r->selections||r->reference||r->runs){r->errors++;return;}
        r->prefix=integer(line,"prefix");r->ctx=integer(line,"ctx");r->warmup=integer(line,"warmup");r->blocks=integer(line,"repeats");
        r->vocab=integer(line,"vocab");r->layers=integer(line,"layers");r->depth=integer(line,"mtp_depth");
        if(!mode(line)||!is(line,"single_run","yes")||r->blocks!=1||r->vocab!=248320||r->layers!=49||r->depth!=2||
           (r->prefix!=2048&&r->prefix!=8192&&r->prefix!=32768&&r->prefix!=65536)||r->ctx!=r->prefix+257||r->warmup!=16||
           integer(line,"measured")!=128||integer(line,"prefill_chunk")!=8192||!is(line,"include_selection","yes")||
           !is(line,"control","2/32")||!is(line,"candidate","1/32")||!is(line,"candidate_env","none")||
           !is(line,"control_value","unset")||!is(line,"candidate_value","1")||!is(line,"reference","control_mtp_same_depth")||
           !is(line,"drafts","full_vocabulary")||!is(line,"acceptance","production")||!is(line,"unchanged_strict_harness","yes")||
           !field(line,"model",r->model,sizeof(r->model))||!field(line,"prompt",r->prompt,sizeof(r->prompt)))r->errors++;
    }else if(!strncmp(line,"ordinary_selection ",19)){
        if(e->selection_count<129)e->selected[e->selection_count]=integer(line,"mtp_trajectory_id");
        e->selection_count++;consume(r,line);
    }else if(!strncmp(line,"run=",4)){
        if(r->runs){r->errors++;return;}r->runs=1;sample *s=&r->samples[0];double tps=0,mean=0;
        s->variant=0;s->cycles=integer(line,"cycles");
        if(e->terminal||r->aggregate[0]||r->validation[0]!=3||r->validation[1]||!mode(line)||integer(line,"run")!=1||integer(line,"repeat")!=1||
           integer(line,"slot")!=1||!is(line,"variant","control")||!is(line,"pattern","single")||integer(line,"tokens")!=128||
           s->cycles!=r->reference_cycles||!number(line,"seconds",&s->seconds)||s->seconds<=0||
           !number(line,"tokens_per_second",&tps)||!rate_matches(s->seconds,tps,128)||!number(line,"first_cycle_ms",&s->first_ms)||
           !number(line,"mean_token_ms",&mean)||!near(mean,1000*s->seconds/128,0.00000052)||!pending(line)||
           !is(line,"cycle_token_selection_exact","yes")||!is(line,"final_logits_exact","yes")||
           !is(line,"final_snapshot_bytes_exact","yes")||!is(line,"restored_continuation_exact","yes")||
           !is(line,"reference","control_mtp_same_depth"))r->errors++;
        r->n[0]=1;r->seconds[0]=s->seconds;r->first_ms[0]=s->first_ms;
    }else if(!strncmp(line,"single_summary ",15)){
        if(++e->terminal!=1||r->runs!=1||timed->memory!=1||timed->cycle_rows!=timed->cycles||r->aggregate[0]||r->aggregate[1]||!mode(line)||!is(line,"same_path_replay","yes")||!is(line,"candidate_evaluated","no")||
           !is(line,"comparison","external_pairing_required")||!is(line,"ordinary_equivalence","not_claimed")||
           !pending(line)||!is(line,"accepted","no"))r->errors++;
    }else if(!strncmp(line,"paired",6)){r->errors++;}
    else consume(r,line);
    if(strstr(line,"failed:")||strstr(line,"mismatch")||strstr(line,"nonfinite"))r->errors++;
}
static int ext_read(external_sample *e,const char *path){
    report *r=&e->r;r->path=strdup(path);FILE *f=fopen(path,"r");if(!r->path||!f)return 0;
    char *line=NULL;size_t cap=0;ssize_t n;
    while((n=getline(&line,&cap,f))>=0){if(n>32768||memchr(line,0,(size_t)n)){r->errors++;continue;}ext_consume(e,line);}
    if(ferror(f))r->errors++;fclose(f);free(line);
    sample *s=&r->samples[0];
    if(e->provenance!=1||r->config!=1||e->terminal!=1||r->reference!=1||r->refhash!=1||r->nrefs!=r->reference_cycles+1||
       r->nrefs<66||r->nrefs>129||r->warm_cycles<9||r->warm_cycles>16||r->selections!=129||e->selection_count!=129||r->evidence!=1||r->drifts!=r->nrefs||
       r->declared_frontiers!=r->nrefs||r->declared_agreements!=r->agreements||r->runs!=1||r->validation[0]!=3||r->validation[1]||
       r->aggregate[0]!=1||r->aggregate[1]||s->hash!=1||s->resources!=1||s->memory!=1||s->process_valid!=1||s->swap_valid!=1||s->vm_valid!=1||
       s->cycle_rows!=s->cycles||s->offset!=128||!isfinite(s->cycle_sum)||s->cycle_sum>s->seconds+s->cycles*0.000000001)r->errors++;
    if(r->nrefs&&r->refs[r->nrefs-1].offset!=128)r->errors++;
    for(int i=0;i<r->nrefs;i++){
        int offset=r->refs[i].offset;
        if(offset<0||offset>128||r->drift_offsets[i]!=offset||r->refs[i].id!=e->selected[offset])r->errors++;
    }
    unsigned char bytes[129*4];for(int i=0;i<129;i++)for(int j=0;j<4;j++)bytes[4*i+j]=(unsigned char)((uint32_t)e->selected[i]>>(8*j));
    char hash[65];sha_bytes(bytes,sizeof(bytes),hash);if(strcmp(hash,r->trajectory))r->errors++;
    char *expected=ext_expected_args(r);if(!expected||strcmp(expected,e->args))r->errors++;free(expected);
    if(!ext_sidecars(e))r->errors++;
    return r->valid=!r->errors;
}
static int ext_equal(const external_sample *a,const external_sample *b){
    const report *x=&a->r,*y=&b->r;
    if(x->prefix!=y->prefix||x->depth!=y->depth||strcmp(x->model,y->model)||strcmp(x->prompt,y->prompt)||strcmp(a->args,b->args)||x->nrefs!=y->nrefs||
       strcmp(x->trajectory,y->trajectory)||strcmp(x->plan,y->plan))return 0;
    for(int i=0;i<x->nrefs;i++){const frontier *p=&x->refs[i],*q=&y->refs[i];
        if(p->offset!=q->offset||p->n!=q->n||p->id!=q->id||p->bytes!=q->bytes||strcmp(p->state,q->state)||strcmp(p->logits,q->logits))return 0;}
    return 1;
}
static int ext_add(unsigned long long *a,unsigned long long b){if(ULLONG_MAX-*a<b)return 0;*a+=b;return 1;}
static void ext_stats(external_case *c){
    c->valid=c->count>=12&&c->count%4==0;c->blocks=c->count/4;c->gain=c->low=c->high=NAN;
    double seconds[2]={0},ratios[32],sum=0,variance=0;int counts[2]={0};unsigned long long resources[2][4]={{0}};
    for(int i=0;i<c->count;i++){external_sample *s=c->runs[i];int v=(i/4%2?"BAAB":"ABBA")[i%4]=='B';
        if(!s||!s->r.valid||s->variant!=v||!c->runs[0]||!ext_equal(c->runs[0],s)){c->valid=0;continue;}
        const sample *m=&s->r.samples[0];
        if(!ext_add(&resources[v][0],m->reads)||!ext_add(&resources[v][1],m->writes)||
           !ext_add(&resources[v][2],m->swapins)||!ext_add(&resources[v][3],m->swapouts))c->valid=0;
        counts[v]++;seconds[v]+=s->r.samples[0].seconds;
    }
    for(int v=0;v<2;v++)c->tps[v]=seconds[v]>0?128*counts[v]/seconds[v]:NAN;
    if(!c->valid)return;
    for(int b=0;b<c->blocks;b++){double a[2]={0};for(int j=0;j<4;j++){external_sample *s=c->runs[b*4+j];a[s->variant]+=s->r.samples[0].seconds;}
        ratios[b]=log(a[0])-log(a[1]);sum+=ratios[b];}
    double mean=sum/c->blocks;for(int b=0;b<c->blocks;b++){double d=ratios[b]-mean;variance+=d*d;}
    int df=c->blocks-1;double radius=critical[df<30?df:30]*sqrt(variance/df/c->blocks);
    c->gain=100*expm1(mean);c->low=100*expm1(mean-radius);c->high=100*expm1(mean+radius);
    if(!isfinite(c->gain)||!isfinite(c->low)||!isfinite(c->high))c->valid=0;
}
static void ext_metrics(FILE *f,const external_case *c,int v){
    if(!c->valid){for(int i=0;i<12;i++)fputc(',',f);return;}
    double seconds=0,first=0,cycles=0;unsigned n=0;unsigned long long rss=0,foot=0,reads=0,writes=0,swap=0,si=0,so=0;
    for(int i=0;i<c->count;i++){const external_sample *e=c->runs[i];if(!e||e->variant!=v||!e->r.valid)continue;const sample *s=&e->r.samples[0];
        n++;seconds+=s->seconds;first+=s->first_ms;cycles+=s->cycles;if(s->resident>rss)rss=s->resident;if(s->footprint>foot)foot=s->footprint;
        reads+=s->reads;writes+=s->writes;if(s->swap>swap)swap=s->swap;si+=s->swapins;so+=s->swapouts;
    }
    fprintf(f,",%.9g,%.9g,%.9g,%.9g,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu",n?seconds*1000/n:NAN,n?seconds*1000/(128*n):NAN,n?first/n:NAN,n?cycles/n:NAN,n,rss,foot,reads,writes,swap,si,so);
}
static int ext_decimal(const char *text,int maximum){
    if(!*text)return -1;for(const char *p=text;*p;p++)if(!isdigit((unsigned char)*p))return -1;
    errno=0;char *end;unsigned long value=strtoul(text,&end,10);return errno||*end||value<1||value>(unsigned)maximum?-1:(int)value;
}
int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: %s MANIFEST.tsv NEW.csv NEW.svg\n",argv[0]);return 2;}
    FILE *manifest=fopen(argv[1],"r");if(!manifest)return 2;external_case cases[EXT_CASES]={0};int count=0,bad=0;char *line=NULL;size_t cap=0;
    const external_sample *build[2]={0};
    while(getline(&line,&cap,manifest)>=0){if(!strncmp(line,"case\t",5))continue;char name[128],variant_name[32],path[4096],block_text[32],slot_text[32];int block,slot;
        if(sscanf(line,"%127[^\t]\t%31[^\t]\t%31[^\t]\t%31[^\t]\t%4095[^\n]",name,block_text,slot_text,variant_name,path)!=5||
           (block=ext_decimal(block_text,32))<1||(slot=ext_decimal(slot_text,4))<1||
           (strcmp(variant_name,"control")&&strcmp(variant_name,"candidate"))){bad=1;break;}
        int k=0;for(;k<count;k++)if(!strcmp(name,cases[k].name))break;if(k==count){if(count==EXT_CASES){bad=1;break;}strcpy(cases[count++].name,name);}
        external_case *c=&cases[k];int index=(block-1)*4+slot-1;if(c->runs[index]){bad=1;break;}
        external_sample *e=calloc(1,sizeof(*e));if(!e){bad=1;break;}c->runs[index]=e;e->present=1;e->variant=!strcmp(variant_name,"candidate");
        if(!ext_read(e,path)){fprintf(stderr,"Invalid fixed-depth MTP single-run evidence: %s\n",path);bad=1;}
        const char *corpus=!strcmp(base_name(e->r.prompt),"original-code.txt")?"code":!strcmp(base_name(e->r.prompt),"promessi_sposi.txt")?"prose":NULL;
        char expected_case[128];snprintf(expected_case,sizeof(expected_case),"mtp-depth2-%s-%d",corpus?corpus:"invalid",e->r.prefix);
        if(!corpus||strcmp(c->name,expected_case)||(build[0]&&strcmp(build[0]->r.model,e->r.model))){e->r.valid=0;bad=1;}
        const external_sample *b=build[e->variant];if(!b)build[e->variant]=e;else if(strcmp(b->runtime,e->runtime)||strcmp(b->binary,e->binary)||strcmp(b->settings,e->settings)){e->r.valid=0;bad=1;}
        if(index+1>c->count)c->count=index+1;
    }
    if(ferror(manifest)||!count)bad=1;free(line);fclose(manifest);for(int k=0;k<count;k++){ext_stats(&cases[k]);if(!cases[k].valid)bad=1;}
    if(same_file(argv[1],argv[2])||same_file(argv[1],argv[3])||same_file(argv[2],argv[3]))return 2;
    int a=open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644),b=a<0?-1:open(argv[3],O_WRONLY|O_CREAT|O_EXCL,0644);
    if(a<0||b<0){if(a>=0){close(a);unlink(argv[2]);}return 2;}FILE *csv=fdopen(a,"w"),*svg=fdopen(b,"w");if(!csv||!svg)return 2;
    fputs("case,mode,baseline_kind,fixed_depth,status,accepted,blocks,process_runs,original_tps,candidate_tps,paired_gain_percent,low95_percent,high95_percent,all_public_logits_sha256_exact,all_public_full_snapshot_sha256_exact,trajectory_and_cycle_plan_exact,timed_final_hashes_exact",csv);
    const char *metrics[]={"mean_timed_region_ms","mean_committed_token_ms","mean_first_mtp_cycle_ms","mean_cycles","valid_resource_rows","max_resident_bytes","max_footprint_bytes","timed_disk_read_bytes","timed_disk_write_bytes","max_global_swap_used_bytes","timed_swapins_pages","timed_swapouts_pages"};
    for(int v=0;v<2;v++)for(unsigned i=0;i<sizeof(metrics)/sizeof(metrics[0]);i++)fprintf(csv,",%s_%s",v?"candidate":"original",metrics[i]);
    fputs(",ordinary_equivalence,quality_gate,rejection_recovery_gate,latency_scope,resident_frontier,io_elapsed_time\n",csv);
    fprintf(svg,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1120\" height=\"%d\"><title>Original versus candidate fixed-depth2 MTP</title><rect width=\"100%%\" height=\"100%%\" fill=\"white\"/><g font-family=\"Arial,sans-serif\" fill=\"#173346\"><text x=\"20\" y=\"30\" font-size=\"22\">Original versus candidate — fixed-depth2 MTP</text><text x=\"20\" y=\"55\" font-size=\"13\">External ABBA; full public-frontier hashes. Adaptive production policy is not measured.</text>",110+count*48);
    double lo=-5,hi=5;for(int i=0;i<count;i++)if(cases[i].valid){lo=fmin(lo,cases[i].low);hi=fmax(hi,cases[i].high);}
    for(int i=0;i<count;i++){external_case *c=&cases[i];const char *status=!c->valid?"invalid_evidence":c->low>0?"positive_paired_ci":c->high<0?"negative_paired_ci":"inconclusive";
        fprintf(csv,"%s,mtp_same_depth,original,2,%s,no,%d,%d,",c->name,status,c->blocks,c->count);
        if(c->valid)fprintf(csv,"%.9g,%.9g,",c->tps[0],c->tps[1]);else fputs(",,",csv);
        if(c->valid)fprintf(csv,"%.9g,%.9g,%.9g,yes,yes,yes,yes",c->gain,c->low,c->high);else fputs(",,,not_established,not_established,not_established,not_established",csv);
        ext_metrics(csv,c,0);ext_metrics(csv,c,1);fputs(",not_claimed,external_required,external_required,warm_MTP_cycle_including_next_draft_and_selection,after_recovery_and_pool_drain,not_collected\n",csv);
        int y=90+i*48;fprintf(svg,"<text x=\"20\" y=\"%d\" font-size=\"13\">",y);xml(svg,c->name);fputs("</text>",svg);
        if(c->valid)fprintf(svg,"<path d=\"M%.2f %d H%.2f\" stroke=\"#9b711e\" stroke-width=\"2\"/><circle cx=\"%.2f\" cy=\"%d\" r=\"4\" fill=\"#9b711e\"/><text x=\"660\" y=\"%d\" font-size=\"12\">%+.3f%% [%+.3f, %+.3f]</text>",660+400*(c->low-lo)/(hi-lo),y-5,660+400*(c->high-lo)/(hi-lo),660+400*(c->gain-lo)/(hi-lo),y-5,y+14,c->gain,c->low,c->high);
        else fprintf(svg,"<text x=\"660\" y=\"%d\">Invalid evidence</text>",y);
    }
    fputs("</g></svg>\n",svg);if(ferror(csv)||ferror(svg))bad=1;if(fclose(csv)||fclose(svg))bad=1;
    for(int i=0;i<count;i++)for(int j=0;j<cases[i].count;j++)if(cases[i].runs[j]){free((void*)cases[i].runs[j]->r.path);free(cases[i].runs[j]);}
    return bad?1:0;
}
