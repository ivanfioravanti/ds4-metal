/* Cross-process ABBA analysis. This accepts only warmed --single-run logs
 * with native launcher provenance and identical full-logit/checkpoint hashes. */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_CASES=128, MAX_RUNS=128 };
typedef struct {
    char logits[65], checkpoint[65];
    unsigned long long bytes;
    long position;
} fingerprint;
typedef struct {
    char runtime[65], binary[65], protocols[2048], args[12000];
    fingerprint hashes[3];
    unsigned hash_mask;
    int variant, present, valid;
    long tokens, prefix, first_token_id;
    double seconds, first_cycle_ms, first_token_ms, selection_ms;
    int telemetry, process_valid, swap_valid, vm_valid;
    unsigned long long resident, footprint, reads, writes, swap_used, swapin, swapout;
    char path[4096];
} sample;
typedef struct {
    char name[128], mode[32], error[128];
    sample *samples;
    int maxrun, blocks, valid;
    double tps[2], gain, low, high;
} testcase;
static const double critical[] = {
    0,12.706205,4.302653,3.182446,2.776445,2.570582,2.446912,2.364624,
    2.306004,2.262157,2.228139,2.200985,2.178813,2.160369,2.144787,
    2.131450,2.119906,2.109816,2.100922,2.093024,2.085964,2.079614,
    2.073874,2.068658,2.063899,2.059539,2.055529,2.051831,2.048407,
    2.045230,2.042273
};
static int token(const char *line,const char *key,char *out,size_t cap) {
    char needle[128];snprintf(needle,sizeof(needle),"%s=",key);
    const char *p=line;
    while((p=strstr(p,needle))) {
        if(p!=line&&!isspace((unsigned char)p[-1])) { p++;continue; }
        p+=strlen(needle);size_t n=strcspn(p," \t\r\n");
        if(!n||n>=cap)return 0;memcpy(out,p,n);out[n]=0;return 1;
    }
    return 0;
}
static long integer(const char *line,const char *key,long fallback) {
    char s[64],*end;if(!token(line,key,s,sizeof(s)))return fallback;
    long v=strtol(s,&end,10);return *end?fallback:v;
}
static int digest(const char *s) {
    if(strlen(s)!=64)return 0;
    for(int i=0;i<64;i++)if(!isxdigit((unsigned char)s[i]))return 0;
    return 1;
}
static int json_hash(const char *line,const char *key,char out[65]) {
    char find[128];snprintf(find,sizeof(find),"\"%s\":\"",key);
    const char *p=strstr(line,find);if(!p)return 0;p+=strlen(find);
    if(strlen(p)<65||p[64]!='"')return 0;
    memcpy(out,p,64);out[64]=0;return digest(out);
}
static int nonnegative_number(const char *line,const char *key,double *value) {
    char text[64],*end;
    if(!token(line,key,text,sizeof(text)))return 0;
    *value=strtod(text,&end);
    return !*end&&isfinite(*value)&&*value>=0;
}
static int read_sample(sample *s,const char *mode) {
    FILE *f=fopen(s->path,"r");if(!f) { perror(s->path);return 0; }
    char *line=NULL;size_t cap=0;int config=0,provenance=0,runs=0,terminal=0,bad=0;
    const int prefill=!strcmp(mode,"prefill");
    s->first_cycle_ms=s->first_token_ms=s->selection_ms=NAN;
    s->first_token_id=-1;
    while(getline(&line,&cap,f)>=0) {
        if(!strncmp(line,"build_provenance=",17)) {
            char expected[40];snprintf(expected,sizeof(expected),"\"runtime\":\"%s\"",s->variant?"candidate":"original");
            const char *p=strstr(line,"\"protocol_source_hashes\":{"),*q;
            if(!strstr(line,expected)||!json_hash(line,"runtime_files_sha256",s->runtime)||
                !json_hash(line,"binary_sha256",s->binary)||!p||(q=strchr(p,'}'))==NULL||
                (size_t)(q-p)>=sizeof(s->protocols)) { bad=1;continue; }
            memcpy(s->protocols,p,(size_t)(q-p));s->protocols[q-p]=0;
            p=strstr(line,"\"command_args\":[");
            if(!p||strlen(p)>=sizeof(s->args)) { bad=1;continue; }
            snprintf(s->args,sizeof(s->args),"%s",p);s->args[strcspn(s->args,"\r\n")]=0;
            provenance++;
        }
        if(strstr(line,"single_run=yes")&&(strstr(line,"configuration mode=")||
            strstr(line,"metal-prefill-variant-bench: model="))) {
            char observed_mode[32]="prefill";
            if(!prefill&&!token(line,"mode",observed_mode,sizeof(observed_mode)))bad=1;
            if(strcmp(mode,observed_mode))bad=1;
            s->prefix=integer(line,"prefix",-1);
            if(prefill&&!strstr(line,"compare_state=yes"))bad=1;
            if(!prefill&&strcmp(mode,"mtp_greedy")&&strcmp(mode,"ordinary"))bad=1;
            config++;
        }
        if(!strncmp(line,"hashes run=",11)) {
            long step=integer(line,"continuation",0),run=integer(line,"run",-1);
            if(run!=1||step<0||step>2||(s->hash_mask&(1u<<step))) { bad=1;continue; }
            fingerprint *h=&s->hashes[step];char bytes[64],*end;
            h->position=integer(line,"position",-1);
            if(!token(line,"logits_sha256",h->logits,sizeof(h->logits))||
                !token(line,"checkpoint_sha256",h->checkpoint,sizeof(h->checkpoint))||
                !digest(h->logits)||!digest(h->checkpoint)||
                !token(line,"checkpoint_bytes",bytes,sizeof(bytes))) { bad=1;continue; }
            h->bytes=strtoull(bytes,&end,10);
            if(*end||!h->bytes||h->position<0) { bad=1;continue; }
            s->hash_mask|=1u<<step;
        }
        if(!strncmp(line,"run=",4)&&strstr(line," pattern=")) {
            char variant[32],pattern[32],seconds[64],observed_mode[32]="prefill",*end;
            if(integer(line,"run",-1)!=1||integer(line,"slot",-1)!=1||
                !token(line,"variant",variant,sizeof(variant))||strcmp(variant,"control")||
                !token(line,"pattern",pattern,sizeof(pattern))||strcmp(pattern,"single")||
                !token(line,"seconds",seconds,sizeof(seconds))) { bad=1;continue; }
            if(!prefill&&!token(line,"mode",observed_mode,sizeof(observed_mode)))bad=1;
            if(strcmp(observed_mode,mode))bad=1;
            s->seconds=strtod(seconds,&end);s->tokens=integer(line,"tokens",-1);
            if(*end||!isfinite(s->seconds)||s->seconds<=0||s->tokens<=0)bad=1;
            if(token(line,"first_cycle_ms",seconds,sizeof(seconds))) {
                s->first_cycle_ms=strtod(seconds,&end);
                if(*end||!isfinite(s->first_cycle_ms)||s->first_cycle_ms<0)bad=1;
            }
            if(prefill) {
                s->first_token_id=integer(line,"first_token_id",-1);
                if(!nonnegative_number(line,"first_token_ms",&s->first_token_ms)||
                   !nonnegative_number(line,"selection_ms",&s->selection_ms)||
                   s->first_token_id<0||s->first_token_id>2147483647L||
                   /* V3 prints seconds and milliseconds to nine decimals.
                    * Allow two nanoseconds for decimal rounding only. */
                   fabs(s->first_token_ms-1000*s->seconds-s->selection_ms)>0.000002)bad=1;
            } else if(!isfinite(s->first_cycle_ms))bad=1;
            runs++;
        }
        if(!strncmp(line,"telemetry run=",14)||!strncmp(line,"memory run=",11)) {
            if(integer(line,"run",-1)!=1||s->telemetry++)bad=1;
            s->process_valid=integer(line,"process_valid",0)==1;
            s->swap_valid=integer(line,"swap_valid",0)==1;
            s->vm_valid=integer(line,"vm_valid",0)==1;
            s->resident=(unsigned long long)integer(line,"resident_bytes",0);
            s->footprint=(unsigned long long)integer(line,"footprint_bytes",0);
            s->reads=(unsigned long long)integer(line,"disk_read_bytes",0);
            s->writes=(unsigned long long)integer(line,"disk_write_bytes",0);
            s->swap_used=(unsigned long long)integer(line,"global_swap_used_bytes",0);
            s->swapin=(unsigned long long)integer(line,prefill?"global_swapin_pages":"global_swapins_pages",0);
            s->swapout=(unsigned long long)integer(line,prefill?"global_swapout_pages":"global_swapouts_pages",0);
        }
        if(!strncmp(line,"aggregate ",10)&&strstr(line,"variant=control")&&integer(line,"runs",-1)==1)terminal++;
        if(!strncmp(line,"paired ",7)||!strncmp(line,"paired_ci ",10))bad=1;
        if(strstr(line,"checkpoint mismatch")||strstr(line,"raw logit mismatch")||strstr(line,"nonfinite logit"))bad=1;
    }
    if(ferror(f))bad=1;
    free(line);fclose(f);
    if(provenance!=1||config!=1||runs!=1||terminal!=1||s->hash_mask!=(prefill?7u:1u)||s->prefix<1)bad=1;
    for(int i=0;i<(prefill?3:1);i++)
        if(s->hashes[i].position!=(prefill?s->prefix+i:s->prefix+s->tokens))bad=1;
    return !bad;
}
static void reject(testcase *c,const char *why) {
    if(!c->error[0])snprintf(c->error,sizeof(c->error),"%s",why);
    c->valid=0;
}
static void evaluate(testcase *c) {
    c->valid=1;c->gain=c->low=c->high=NAN;
    if(c->maxrun<12||c->maxrun%4)reject(c,"incomplete_external_blocks");
    c->blocks=c->maxrun/4;
    double secs[2]={0},tokens[2]={0},ratios[MAX_RUNS/4]={0},sum=0;
    sample *ref=&c->samples[0],*buildref[2]={NULL,NULL};
    for(int i=0;i<c->maxrun;i++) {
        sample *s=&c->samples[i];
        if(!s->present||!s->valid) { reject(c,"missing_or_invalid_single_run");continue; }
        int expected=((i/4)%2 ? "BAAB" : "ABBA")[i%4]=='B';
        if(s->variant!=expected)reject(c,"invalid_external_order");
        if(s->tokens!=ref->tokens||s->prefix!=ref->prefix||strcmp(s->args,ref->args)||
            strcmp(s->protocols,ref->protocols))reject(c,"incompatible_protocol_or_arguments");
        if(!strcmp(c->mode,"prefill")&&s->first_token_id!=ref->first_token_id)
            reject(c,"first_selected_token_mismatch");
        if(s->hash_mask!=ref->hash_mask)reject(c,"missing_hash_frontier");
        for(int h=0;h<3;h++)if(s->hash_mask&(1u<<h)) {
            fingerprint *a=&s->hashes[h],*b=&ref->hashes[h];
            if(a->position!=b->position||a->bytes!=b->bytes||strcmp(a->logits,b->logits)||
                strcmp(a->checkpoint,b->checkpoint))reject(c,"cross_build_hash_mismatch");
        }
        if(!buildref[s->variant])buildref[s->variant]=s;
        else if(strcmp(s->runtime,buildref[s->variant]->runtime)||strcmp(s->binary,buildref[s->variant]->binary))
            reject(c,"runtime_or_binary_changed_within_case");
        secs[s->variant]+=s->seconds;tokens[s->variant]+=s->tokens;
    }
    for(int v=0;v<2;v++)c->tps[v]=secs[v]>0?tokens[v]/secs[v]:NAN;
    if(!c->valid)return;
    for(int b=0;b<c->blocks;b++) {
        double a=0,d=0;
        for(int j=0;j<4;j++) { sample *s=&c->samples[b*4+j];if(s->variant)d+=s->seconds;else a+=s->seconds; }
        ratios[b]=log(a/d);sum+=ratios[b];
    }
    double mean=sum/c->blocks,var=0;
    for(int b=0;b<c->blocks;b++) { double x=ratios[b]-mean;var+=x*x; }
    int df=c->blocks-1;
    double radius=critical[df<30?df:30]*sqrt(var/df/c->blocks);
    c->gain=100*expm1(mean);c->low=100*expm1(mean-radius);c->high=100*expm1(mean+radius);
}
static void escaped(FILE *f,const char *s) {
    for(;*s;s++)if(*s=='&')fputs("&amp;",f);else if(*s=='<')fputs("&lt;",f);
        else if(*s=='>')fputs("&gt;",f);else if(*s=='"')fputs("&quot;",f);else fputc(*s,f);
}
static void metric_fields(FILE *f,const testcase *c,int variant) {
    int n=0,first_n=0,process_n=0,swap_n=0,vm_n=0;
    double seconds=0,tokens=0,first=0,first_token=0,selection=0;
    int prefill=!strcmp(c->mode,"prefill");
    unsigned long long resident=0,footprint=0,reads=0,writes=0,swap=0,swapin=0,swapout=0;
    for(int i=0;i<c->maxrun;i++) {
        const sample *s=&c->samples[i];if(!s->present||!s->valid||s->variant!=variant)continue;
        n++;seconds+=s->seconds;tokens+=s->tokens;
        if(isfinite(s->first_cycle_ms)) { first_n++;first+=s->first_cycle_ms; }
        if(prefill) { first_token+=s->first_token_ms;selection+=s->selection_ms; }
        if(s->process_valid) {
            process_n++;if(s->resident>resident)resident=s->resident;if(s->footprint>footprint)footprint=s->footprint;
            reads+=s->reads;writes+=s->writes;
        }
        if(s->swap_valid) { swap_n++;if(s->swap_used>swap)swap=s->swap_used; }
        if(s->vm_valid) { vm_n++;swapin+=s->swapin;swapout+=s->swapout; }
    }
    fprintf(f,",%.6f,",n?1000*seconds/n:NAN);
    if(prefill&&tokens)fprintf(f,"%.6f",1000*seconds/tokens);
    fputc(',',f);
    if(!prefill&&tokens)fprintf(f,"%.6f",1000*seconds/tokens);
    fputc(',',f);
    if(prefill&&n)fprintf(f,"%.6f",first_token/n);
    fputc(',',f);
    if(prefill&&n)fprintf(f,"%.6f",selection/n);
    fputc(',',f);
    if(!strcmp(c->mode,"ordinary")&&first_n)fprintf(f,"%.6f",first/first_n);
    fputc(',',f);
    if(!strcmp(c->mode,"mtp_greedy")&&first_n)fprintf(f,"%.6f",first/first_n);
    fprintf(f,",%d,",process_n);
    if(process_n)fprintf(f,"%llu,%llu,%llu,%llu",resident,footprint,reads,writes);else fputs(",,,",f);
    fprintf(f,",%d,",swap_n);if(swap_n)fprintf(f,"%llu",swap);
    fprintf(f,",%d,",vm_n);if(vm_n)fprintf(f,"%llu,%llu",swapin,swapout);else fputc(',',f);
}
static void outputs(FILE *csv,FILE *svg,testcase *cs,int n) {
    fputs("case,mode,status,accepted,external_blocks,process_runs,original_tps,candidate_tps,paired_gain_percent,low95_percent,high95_percent,full_logits_sha256_exact,checkpoints_sha256_exact",csv);
    const char *metric_names[]={"mean_timed_region_ms","mean_prompt_token_ms","mean_committed_token_ms","mean_first_selected_token_ms","mean_initial_selection_ms","mean_first_decode_step_ms","mean_first_mtp_cycle_ms","valid_process_rows","max_resident_bytes","max_footprint_bytes","timed_disk_read_bytes","timed_disk_write_bytes","valid_swap_rows","max_global_swap_used_bytes","valid_vm_rows","timed_global_swapin_pages","timed_global_swapout_pages"};
    for(int v=0;v<2;v++)for(size_t m=0;m<sizeof(metric_names)/sizeof(metric_names[0]);m++)
        fprintf(csv,",%s_%s",v?"candidate":"original",metric_names[m]);
    fputs(",latency_scope,resident_sample_frontier,io_elapsed_time,quality_gate,coverage_gate\n",csv);
    double lo=-5,hi=5;
    for(int i=0;i<n;i++)if(cs[i].valid) { lo=fmin(lo,cs[i].low);hi=fmax(hi,cs[i].high); }
    lo=floor(lo/5)*5;hi=ceil(hi/5)*5;int height=130+n*45;
    fprintf(svg,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1150\" height=\"%d\" role=\"img\"><title>Original versus candidate: external process ABBA</title><rect width=\"100%%\" height=\"100%%\" fill=\"white\"/><g font-family=\"Arial,sans-serif\" fill=\"#173346\"><text x=\"20\" y=\"30\" font-size=\"23\">Original versus candidate — external process ABBA</text><text x=\"20\" y=\"55\" font-size=\"13\">Strict full-logit and checkpoint hashes. Timing evidence only; no automatic acceptance.</text><text x=\"410\" y=\"83\" font-size=\"12\">Original / candidate tokens/s</text><text x=\"715\" y=\"83\" font-size=\"12\">Paired gain with 95%% CI</text>",height);
    double zero=715+370*(-lo)/(hi-lo);
    fprintf(svg,"<path d=\"M%.2f 93 V%d\" stroke=\"#c2ccd4\"/>",zero,height-28);
    for(int i=0;i<n;i++) {
        testcase *c=&cs[i];const char *status=c->valid ? c->low>0?"positive_paired_ci":c->high<0?"negative_paired_ci":"inconclusive" : c->error;
        fprintf(csv,"%s,%s,%s,no,%d,%d,%.6f,%.6f,",c->name,c->mode,status,c->blocks,c->maxrun,c->tps[0],c->tps[1]);
        if(c->valid)fprintf(csv,"%.6f,%.6f,%.6f,yes,yes",c->gain,c->low,c->high);
        else fputs(",,,not_established,not_established",csv);
        metric_fields(csv,c,0);metric_fields(csv,c,1);
        fprintf(csv,",%s,%s,not_collected,not_checked,not_checked\n",
                !strcmp(c->mode,"prefill")?"warm_sync_then_immediate_initial_argmax":
                !strcmp(c->mode,"ordinary")?"warm_decode_step_then_next_argmax":"warm_mtp_cycle_return_including_next_draft_and_argmax",
                !strcmp(c->mode,"prefill")?"after_initial_selection":"after_pool_drain");
        int y=110+i*45;fprintf(svg,"<text x=\"20\" y=\"%d\" font-size=\"12\">",y);escaped(svg,c->name);fputs("</text>",svg);
        fprintf(svg,"<text x=\"410\" y=\"%d\" font-size=\"12\">%.1f / %.1f</text>",y,c->tps[0],c->tps[1]);
        if(c->valid) {
            double a=715+370*(c->low-lo)/(hi-lo),b=715+370*(c->high-lo)/(hi-lo),m=715+370*(c->gain-lo)/(hi-lo);
            fprintf(svg,"<path d=\"M%.2f %d H%.2f\" stroke=\"#167c91\" stroke-width=\"2\"/><circle cx=\"%.2f\" cy=\"%d\" r=\"4\" fill=\"#167c91\"/><text x=\"715\" y=\"%d\" font-size=\"11\">%+.2f%% [%+.2f, %+.2f]</text>",a,y-5,b,m,y-5,y+12,c->gain,c->low,c->high);
        } else { fprintf(svg,"<text x=\"715\" y=\"%d\" font-size=\"11\" fill=\"#b74729\">",y);escaped(svg,status);fputs("</text>",svg); }
    }
    fputs("</g></svg>\n",svg);
}
int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"usage: %s MANIFEST.tsv OUTPUT.csv OUTPUT.svg\nmanifest columns: case mode block slot variant log (tab-separated; variant control/candidate)\n",argv[0]);return 2; }
    FILE *f=fopen(argv[1],"r");if(!f) { perror(argv[1]);return 2; }
    testcase *cs=calloc(MAX_CASES,sizeof(*cs));if(!cs)return 2;
    int n=0,bad=0;char *line=NULL;size_t cap=0;
    while(getline(&line,&cap,f)>=0) {
        if(line[0]=='#'||!strncmp(line,"case\t",5)||line[0]=='\n')continue;
        char name[128],mode[32],variant[32],path[4096];int block,slot;
        if(sscanf(line,"%127[^\t]\t%31[^\t]\t%d\t%d\t%31[^\t]\t%4095[^\n]",name,mode,&block,&slot,variant,path)!=6||
            block<1||block>MAX_RUNS/4||slot<1||slot>4||
            (strcmp(mode,"prefill")&&strcmp(mode,"ordinary")&&strcmp(mode,"mtp_greedy"))||
            (strcmp(variant,"control")&&strcmp(variant,"candidate"))) { fprintf(stderr,"invalid manifest row\n");bad=1;break; }
        int c=0;for(;c<n;c++)if(!strcmp(cs[c].name,name))break;
        if(c==n) {
            if(n>=MAX_CASES) { bad=1;break; }n++;
            snprintf(cs[c].name,sizeof(cs[c].name),"%s",name);snprintf(cs[c].mode,sizeof(cs[c].mode),"%s",mode);
            cs[c].samples=calloc(MAX_RUNS,sizeof(sample));if(!cs[c].samples) { bad=1;break; }
        } else if(strcmp(cs[c].mode,mode)) { bad=1;break; }
        int index=(block-1)*4+slot-1;sample *s=&cs[c].samples[index];
        if(s->present) { fprintf(stderr,"duplicate manifest slot\n");bad=1;break; }
        s->present=1;s->variant=!strcmp(variant,"candidate");snprintf(s->path,sizeof(s->path),"%s",path);
        s->valid=read_sample(s,mode);if(!s->valid)fprintf(stderr,"invalid single-run evidence: %s\n",path);
        if(index+1>cs[c].maxrun)cs[c].maxrun=index+1;
    }
    if(ferror(f))bad=1;fclose(f);free(line);
    if(!n)bad=1;
    for(int i=0;i<n;i++) { evaluate(&cs[i]);if(!cs[i].valid)bad=1; }
    FILE *csv=fopen(argv[2],"w"),*svg=fopen(argv[3],"w");
    if(!csv||!svg) { perror("output");return 2; }
    outputs(csv,svg,cs,n);
    if(ferror(csv)||ferror(svg))bad=1;
    if(fclose(csv))bad=1;
    if(fclose(svg))bad=1;
    for(int i=0;i<n;i++)free(cs[i].samples);free(cs);
    return bad?1:0;
}
