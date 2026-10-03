/* Separate diagnostic same-depth MTP control experiment.
 * The included strict harness is unchanged and is never the entry point. */
#define main ds4_unchanged_strict_decode_main
#include "metal_decode_schedule_bench.c"
#undef main
#include <fcntl.h>

#ifndef DS4_MTP_SAME_ENTRY
#define DS4_MTP_SAME_ENTRY main
#endif

typedef struct {
    int offset, committed, selected;
    uint64_t snapshot_bytes;
    char snapshot_sha256[65];
} sd_frontier;
typedef struct {
    int steps, cycles, vocab;
    int *tokens;
    float *rows;
    sd_frontier *frontiers;
    ds4_session_snapshot final;
} sd_reference;

static void sd_free(sd_reference *r) {
    free(r->tokens); free(r->rows); free(r->frontiers);
    ds4_session_snapshot_free(&r->final);
    memset(r, 0, sizeof(*r));
}

/* Only source-audited scheduling controls whose acceptance still requires
 * exact same-depth arithmetic/state. Numerical Q8/attention reductions and
 * depth/acceptance/draft-vocabulary changes deliberately have no entry. */
static bool sd_control_value(const char *name,const char *value) {
    static const struct { const char *name,*values; } controls[]={
        {"DS4_QWEN4_MTP_GPU_ARGMAX","0,1"},
        {"DS4_QWEN4_MTP_SWAP_RESTORE","0,1"},
        {"DS4_QWEN4_FLUSH_LAYER","0,1,2,4,8"},
        {"DS4_QWEN4_Q4K_MID_NSG","1,2,4,8"},
        {"DS4_QWEN4_Q4K_MID_NR","1,2"},
        {"DS4_QWEN4_MOE_MV_NSG","4,8,16"},
        {"DS4_QWEN4_MOE_MV_NR","1,2,4"},
        {"DS4_QWEN4_MOE_DOWN_PREFETCH","0,1"},
        {"DS4_METAL_PLAIN_MV_NR0","0,2"},
        {"DS4_QWEN4_DECODE_FUSIONS","0,1"},
        {"DS4_QWEN4_ATTN_QUERY_GROUP","4,6,12"},
        {"DS4_QWEN4_ATTN_MERGE_WIDE","0,1"},
        {"DS4_QWEN4_GDN_NSG","1,2,4,8"},
        {"DS4_QWEN4_GDN_FRONT_THREADS","256,512,1024"},
        {"DS4_QWEN4_NGRAM_OVERLAP","0,1"},
    };
    if(!name || !value)return false;
    const size_t length=strlen(value);
    for(size_t i=0;i<sizeof(controls)/sizeof(controls[0]);i++)if(!strcmp(name,controls[i].name)) {
        for(const char *p=controls[i].values;*p;) {
            const char *end=strchr(p,',');const size_t n=end?(size_t)(end-p):strlen(p);
            if(n==length && !memcmp(p,value,n))return true;
            if(!end)break;
            p=end+1;
        }
        return false;
    }
    return false;
}

static int sd_config(const bench_config *cfg) {
    const bool known=sd_control_value(cfg->candidate_env,cfg->control_value) &&
                     sd_control_value(cfg->candidate_env,cfg->candidate_value);
    if (!cfg->mtp || cfg->measured != 128 || (!cfg->single_run && cfg->repeats < 3) || cfg->repeats > 256 ||
        cfg->warmup < 1 || cfg->warmup > 128 || !cfg->include_selection ||
        cfg->prefill_chunk < 3 || cfg->prefill_chunk > 8192 ||
        (!(cfg->single_run && !cfg->candidate_env) && !known) ||
        (!cfg->single_run && (cfg->control.first != cfg->candidate.first || cfg->control.second != cfg->candidate.second)) ||
        (int64_t)cfg->prefix_tokens + cfg->measured + 4 > cfg->ctx) {
        fprintf(stderr,"mtp-same-depth: require --mtp, fixed depth2/3, --tokens128, 3..256 blocks (or --single-run), warmup1..128, selection, chunk3..8192, sufficient context, identical paired schedules and an allowed exact-schedule control/value\n");
        return 1;
    }
    const char *blocked[] = {"DS4_QWEN4_NO_MTP_BATCH", "DS4_QWEN4_NO_FUSE", "DS4_QWEN4_SPEC_TRACE"};
    for (size_t i=0;i<sizeof(blocked)/sizeof(blocked[0]);i++) if (getenv(blocked[i])) {
        fprintf(stderr,"mtp-same-depth: remove %s\n",blocked[i]); return 1;
    }
    return 0;
}

static int sd_snapshot_copy(ds4_session_snapshot *dst, const ds4_session_snapshot *src) {
    if (!src->ptr || !src->len || src->len > SIZE_MAX) return 1;
    uint8_t *p = realloc(dst->ptr, (size_t)src->len);
    if (!p) return 1;
    dst->ptr=p; dst->len=dst->cap=src->len;
    memcpy(dst->ptr,src->ptr,(size_t)src->len); return 0;
}

/* Hash canonical little-endian integers, excluding timing/padding. Accepted
 * IDs include the final selected token; the plan contains every public
 * frontier's offset, committed count, and selection. Work stays untimed. */
static void sd_put_u32(uint8_t *p,uint32_t v) {
    for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));
}
static void sd_reference_hashes(const sd_reference *r,char trajectory[65],char plan[65]) {
    uint8_t tokens[129*4],frontiers[129*3*4];
    for(int i=0;i<=r->steps;i++)sd_put_u32(tokens+(size_t)i*4,(uint32_t)r->tokens[i]);
    for(int i=0;i<=r->cycles;i++) {
        uint8_t *p=frontiers+(size_t)i*12;
        sd_put_u32(p,(uint32_t)r->frontiers[i].offset);
        sd_put_u32(p+4,(uint32_t)r->frontiers[i].committed);
        sd_put_u32(p+8,(uint32_t)r->frontiers[i].selected);
    }
    ds4_bench_sha256(tokens,((size_t)r->steps+1)*4,trajectory);
    ds4_bench_sha256(frontiers,((size_t)r->cycles+1)*12,plan);
}

static int sd_capture_frontier(ds4_session *s, sd_reference *r, int index, int offset,
                               int committed, int eos, ds4_session_snapshot *scratch,
                               char *err, size_t errlen) {
    float *row = r->rows + (size_t)index*r->vocab;
    if (ds4_session_copy_logits(s,row,r->vocab)!=r->vocab ||
        finite_logits(row,r->vocab,offset,"control_mtp") ||
        ds4_session_save_snapshot(s,scratch,err,errlen)) return 1;
    const int selected=ds4_session_argmax_excluding(s,eos);
    if (selected<0 || selected>=r->vocab || selected==eos) return 1;
    sd_frontier *f=&r->frontiers[index];
    f->offset=offset; f->committed=committed; f->selected=selected;
    f->snapshot_bytes=scratch->len;
    ds4_bench_sha256(scratch->ptr,(size_t)scratch->len,f->snapshot_sha256);
    r->tokens[offset]=selected;
    return 0;
}

static int sd_capture(ds4_session *s, sd_reference *r, int steps, int vocab,
                      int eos, int depth, char *err, size_t errlen) {
    sd_free(r); r->steps=steps; r->vocab=vocab;
    r->tokens=calloc((size_t)steps+1,sizeof(*r->tokens));
    r->frontiers=calloc((size_t)steps+1,sizeof(*r->frontiers));
    r->rows=malloc(((size_t)steps+1)*(size_t)vocab*sizeof(float));
    ds4_session_snapshot scratch={0}; int failed=1;
    if (!r->tokens || !r->frontiers || !r->rows) goto done;
    if (sd_capture_frontier(s,r,0,0,0,eos,&scratch,err,errlen)) goto done;
    int done=0;
    while (done<steps) {
        int ids[3]={-1,-1,-1};
        const int cap=steps-done<3?steps-done:3;
        const int n=ds4_session_eval_speculative_argmax(s,r->tokens[done],cap,eos,ids,cap,err,errlen);
        if(n<1 || n>cap || n>depth || (!r->cycles && n!=1) || ids[0]!=r->tokens[done]) {
            snprintf(err,errlen,"invalid control MTP cycle at offset%d count%d",done,n); goto done;
        }
        for(int j=0;j<n;j++) {
            if(ids[j]<0 || ids[j]>=vocab || ids[j]==eos) {
                snprintf(err,errlen,"control committed invalid/EOS token at offset%d",done+j); goto done;
            }
            r->tokens[done+j]=ids[j];
        }
        done+=n; r->cycles++;
        if(sd_capture_frontier(s,r,r->cycles,done,n,eos,&scratch,err,errlen)) goto done;
    }
    if(sd_snapshot_copy(&r->final,&scratch)) goto done;
    failed=0;
done:
    ds4_session_snapshot_free(&scratch); return failed;
}

static int sd_frontier_check(ds4_session *s, const sd_reference *r, int index,
                             float *row, ds4_session_snapshot *scratch,
                             char *err, size_t errlen) {
    const sd_frontier *f=&r->frontiers[index]; char digest[65];
    if(compare_logits(s,r->rows+(size_t)index*r->vocab,row,r->vocab,f->offset) ||
       ds4_session_save_snapshot(s,scratch,err,errlen)) return 1;
    ds4_bench_sha256(scratch->ptr,(size_t)scratch->len,digest);
    if(scratch->len!=f->snapshot_bytes || strcmp(digest,f->snapshot_sha256)) {
        snprintf(err,errlen,"same-depth full snapshot SHA256 mismatch at public frontier%d",f->offset); return 1;
    }
    return 0;
}

/* No logit reads, hashes, snapshots or diagnostic waits when verify=false.
 * Candidate accepted counts must follow the control's exact cycle schedule. */
static int sd_replay(ds4_session *s, const bench_config *cfg, const sd_reference *r,
                     int eos, bool verify, float *row, ds4_session_snapshot *scratch,
                     cycle_sample *samples, char *err, size_t errlen) {
    if(verify && sd_frontier_check(s,r,0,row,scratch,err,errlen)) return 1;
    for(int cycle=0;cycle<r->cycles;cycle++) {
        const sd_frontier *before=&r->frontiers[cycle], *after=&r->frontiers[cycle+1];
        const int cap=r->steps-before->offset<3?r->steps-before->offset:3;
        int ids[3]={-1,-1,-1};
        const double t0=now_sec();
        const int n=ds4_session_eval_speculative_argmax(s,r->tokens[before->offset],cap,eos,ids,cap,err,errlen);
        const int selected=ds4_session_argmax_excluding(s,eos);
        const double seconds=now_sec()-t0;
        if(n!=after->committed || n<1 || n>cfg->mtp_depth || n>cap ||
           memcmp(ids,r->tokens+before->offset,(size_t)n*sizeof(int)) || selected!=after->selected) {
            snprintf(err,errlen,"same-depth cycle/token/selection mismatch at cycle%d offset%d",cycle+1,before->offset); return 1;
        }
        if(samples)samples[cycle]=(cycle_sample){seconds,n};
        if(verify && sd_frontier_check(s,r,cycle+1,row,scratch,err,errlen)) return 1;
    }
    if(verify && compare_snapshot(s,&r->final,scratch,err,errlen)) return 1;
    return 0;
}

static int sd_final_check(ds4_session *s,const sd_reference *r,float *row,
                          ds4_session_snapshot *scratch,char *err,size_t errlen) {
    return compare_logits(s,r->rows+(size_t)r->cycles*r->vocab,row,r->vocab,r->steps) ||
           compare_snapshot(s,&r->final,scratch,err,errlen);
}

/* Ordinary evaluation is teacher-forced along the control-MTP trajectory.
 * Differences are evidence, never a raw-logit tolerance or quality verdict. */
static int sd_ordinary_evidence(ds4_session *s,const sd_reference *r,int eos,float *row,
                                char *err,size_t errlen) {
    int frontier=0,agreements=0;
    for(int offset=0;offset<=r->steps;offset++) {
        if(ds4_session_copy_logits(s,row,r->vocab)!=r->vocab ||
           finite_logits(row,r->vocab,offset,"ordinary_teacher_forced")) return 1;
        const int selected=ds4_session_argmax_excluding(s,eos);
        if(selected<0 || selected>=r->vocab || selected==eos) return 1;
        const bool same=selected==r->tokens[offset]; agreements+=same;
        printf("ordinary_selection offset=%d ordinary_id=%d mtp_trajectory_id=%d agrees=%s evidence_only=yes\n",
               offset,selected,r->tokens[offset],same?"yes":"no");
        if(frontier<=r->cycles && r->frontiers[frontier].offset==offset) {
            const float *other=r->rows+(size_t)frontier*r->vocab;
            double maximum=0,total=0; uint64_t differing=0;
            for(int i=0;i<r->vocab;i++) {
                const double d=fabs((double)row[i]-(double)other[i]);
                if(d>maximum)maximum=d;
                total+=d;
                differing+=float_bits(row[i])!=float_bits(other[i]);
            }
            printf("ordinary_drift public_frontier=%d offset=%d differing_logits=%llu count=%d max_abs=%.12g mean_abs=%.12g interpretation=reported_not_thresholded\n",
                   frontier,offset,(unsigned long long)differing,r->vocab,maximum,total/r->vocab);
            frontier++;
        }
        if(offset<r->steps && ds4_session_eval(s,r->tokens[offset],err,errlen))return 1;
    }
    printf("ordinary_evidence trajectory=teacher_forced_control_mtp selected_agreements=%d selected_comparisons=%d public_frontiers=%d quality=not_decided_by_this_tool full_state_equivalence=not_claimed\n",
           agreements,r->steps+1,frontier);
    return 0;
}

static int sd_recovery(ds4_session *s,const bench_config *cfg,const sd_reference *measured,
                       const sd_reference *recovery,int eos,float *row,
                       ds4_session_snapshot *scratch,char *err,size_t errlen) {
    /* Caller has already compared its actual final payload to control. */
    if(ds4_session_load_snapshot(s,&measured->final,err,errlen) ||
       compare_snapshot(s,&measured->final,scratch,err,errlen)) return 1;
    return sd_replay(s,cfg,recovery,eos,true,row,scratch,NULL,err,errlen);
}

int DS4_MTP_SAME_ENTRY(int argc,char **argv) {
    const bench_config cfg=parse_options(argc,argv);
    if(sd_config(&cfg) || validate_environment(&cfg) || normalize_output_experiment_env() || select_variant(&cfg,0))return 1;
    if(cfg.output_path) {
        int fd=open(cfg.output_path,O_WRONLY|O_CREAT|O_EXCL,0644);
        if(fd<0 || dup2(fd,STDOUT_FILENO)<0 || dup2(fd,STDERR_FILENO)<0) {perror("mtp-same-depth output");return 1;}
        close(fd);setvbuf(stdout,NULL,_IOLBF,0);setvbuf(stderr,NULL,_IOLBF,0);
    }
    char *text=read_text(cfg.prompt_path),err[512]={0}; int rc=1;
    void *outer_pool=ds4_bench_pool_create(),*pool=NULL;
    ds4_engine *engine=NULL;ds4_session *s=NULL;ds4_tokens prompt={0};
    ds4_session_snapshot base={0},scratch={0};sd_reference measured={0},warm={0},recovery={0};
    float *row=NULL;cycle_sample samples[128];
    ds4_engine_options opt={.model_path=cfg.model_path,.backend=DS4_BACKEND_METAL,
        .context_size=cfg.ctx,.prefill_chunk=cfg.prefill_chunk,.power_percent=100,.warm_weights=true,.glm_mtp=true};
    if(!text || ds4_engine_open(&engine,&opt))goto done;
    if(!ds4_engine_is_qwen4(engine) || ds4_engine_mtp_draft_tokens(engine)<=1)goto done;
    const int vocab=ds4_engine_vocab_size(engine),eos=ds4_token_eos(engine);
    const int layers=ds4_engine_layer_count(engine);
    if(cfg.control.first>layers || cfg.control.second>layers ||
       (!cfg.single_run && (cfg.candidate.first>layers || cfg.candidate.second>layers))) {
        snprintf(err,sizeof(err),"split layers must be between 0 and %d",layers);goto done;
    }
    if(vocab<1 || (size_t)vocab>SIZE_MAX/sizeof(float)/129u)goto done;
    row=malloc((size_t)vocab*sizeof(float));if(!row)goto done;
    ds4_tokenize_text(engine,text,&prompt);free(text);text=NULL;
    if(prompt.len<cfg.prefix_tokens)goto done;
    ds4_tokens prefix={.v=prompt.v,.len=cfg.prefix_tokens,.cap=cfg.prefix_tokens};
    if(ds4_session_create(&s,engine,cfg.ctx) || ds4_session_sync(s,&prefix,err,sizeof(err)) ||
       ds4_session_save_snapshot(s,&base,err,sizeof(err)))goto done;
    printf("configuration mode=mtp_same_depth model=%s prompt=%s vocab=%d layers=%d prefix=%d ctx=%d warmup=%d measured=128 repeats=%d prefill_chunk=%u mtp_depth=%d control=%d/%d candidate=%d/%d candidate_env=%s control_value=%s candidate_value=%s include_selection=yes single_run=%s reference=control_mtp_same_depth drafts=full_vocabulary acceptance=production unchanged_strict_harness=yes\n",
        cfg.model_path,cfg.prompt_path,vocab,layers,cfg.prefix_tokens,cfg.ctx,cfg.warmup,cfg.repeats,cfg.prefill_chunk,cfg.mtp_depth,
        cfg.control.first,cfg.control.second,cfg.candidate.first,cfg.candidate.second,
        cfg.candidate_env?cfg.candidate_env:"none",cfg.control_value?cfg.control_value:"unset",
        cfg.candidate_value?cfg.candidate_value:"unset",cfg.single_run?"yes":"no");
    pool=ds4_bench_pool_create();
    if(ds4_session_load_snapshot(s,&base,err,sizeof(err)) || sd_capture(s,&warm,cfg.warmup,vocab,eos,cfg.mtp_depth,err,sizeof(err)) ||
       ds4_session_load_snapshot(s,&base,err,sizeof(err)) || sd_capture(s,&measured,128,vocab,eos,cfg.mtp_depth,err,sizeof(err)) ||
       ds4_session_load_snapshot(s,&measured.final,err,sizeof(err)) || sd_capture(s,&recovery,2,vocab,eos,cfg.mtp_depth,err,sizeof(err)) ||
       ds4_session_load_snapshot(s,&base,err,sizeof(err)) || sd_ordinary_evidence(s,&measured,eos,row,err,sizeof(err)))goto done;
    printf("reference mode=mtp_same_depth measured_cycles=%d public_frontiers=%d warmup_cycles=%d restored_continuation_cycles=%d first_cycle_tokens=1 intermediate_snapshot_identity=full_payload_sha256 final_snapshot_identity=raw_bytes ordinary_quality=pending\n",
           measured.cycles,measured.cycles+1,warm.cycles,recovery.cycles);
    char trajectory_hash[65],plan_hash[65];
    sd_reference_hashes(&measured,trajectory_hash,plan_hash);
    for(int i=0;i<=measured.cycles;i++) {
        char logits_hash[65];
        ds4_bench_sha256(measured.rows+(size_t)i*vocab,(size_t)vocab*sizeof(float),logits_hash);
        printf("reference_frontier mode=mtp_same_depth index=%d offset=%d committed=%d selected=%d snapshot_bytes=%llu snapshot_sha256=%s logits_sha256=%s\n",
            i,measured.frontiers[i].offset,measured.frontiers[i].committed,measured.frontiers[i].selected,
            (unsigned long long)measured.frontiers[i].snapshot_bytes,measured.frontiers[i].snapshot_sha256,logits_hash);
    }
    printf("reference_hashes trajectory_tokens=129 trajectory_sha256=%s cycle_plan_frontiers=%d cycle_plan_sha256=%s integer_encoding=u32_little_endian\n",trajectory_hash,measured.cycles+1,plan_hash);
    ds4_bench_pool_drain(pool);pool=NULL;
    for(int variant=0;variant<(cfg.single_run?1:2);variant++)for(int replay=0;replay<2;replay++) {
        pool=ds4_bench_pool_create();
        if(select_variant(&cfg,variant) || ds4_session_load_snapshot(s,&base,err,sizeof(err)) ||
           sd_replay(s,&cfg,&warm,eos,true,row,&scratch,NULL,err,sizeof(err)) ||
           ds4_session_load_snapshot(s,&base,err,sizeof(err)) ||
           sd_replay(s,&cfg,&measured,eos,true,row,&scratch,NULL,err,sizeof(err)) ||
           sd_recovery(s,&cfg,&measured,&recovery,eos,row,&scratch,err,sizeof(err)))goto done;
        printf("validation mode=mtp_same_depth variant=%s replay=%d all_public_frontier_logits_exact=yes all_public_frontier_full_snapshot_sha256_exact=yes final_snapshot_bytes_exact=yes cycle_token_selection_exact=yes restored_continuation_exact=yes\n",variant?"candidate":"control",replay+1);
        ds4_bench_pool_drain(pool);pool=NULL;
    }
    double seconds[2]={0},paired_sum=0,paired_sq=0;int runs=0;
    for(int block=0;block<cfg.repeats;block++) {
        const int abba[]={0,1,1,0},baab[]={1,0,0,1};const int *order=block&1?baab:abba;
        double block_seconds[2]={0};
        for(int slot=0;slot<(cfg.single_run?1:4);slot++) {
            const int variant=cfg.single_run?0:order[slot];pool=ds4_bench_pool_create();
            if(select_variant(&cfg,variant) || ds4_session_load_snapshot(s,&base,err,sizeof(err)) ||
               sd_replay(s,&cfg,&warm,eos,false,row,&scratch,NULL,err,sizeof(err)) ||
               ds4_session_load_snapshot(s,&base,err,sizeof(err)))goto done;
            struct rusage before={0},after={0};ds4_bench_memory mb={0},ma={0},md={0};
            ds4_bench_memory_sample(getpid(),&mb);getrusage(RUSAGE_SELF,&before);
            const double start=now_sec();
            if(sd_replay(s,&cfg,&measured,eos,false,row,&scratch,samples,err,sizeof(err)))goto done;
            const double elapsed=now_sec()-start;
            getrusage(RUSAGE_SELF,&after);ds4_bench_memory_sample(getpid(),&ma);
            if(ds4_session_pos(s)!=cfg.prefix_tokens+128 || sd_final_check(s,&measured,row,&scratch,err,sizeof(err)))goto done;
            char state_hash[65],logits_hash[65];const uint64_t snapshot_bytes=scratch.len;
            ds4_bench_sha256(scratch.ptr,(size_t)scratch.len,state_hash);
            ds4_bench_sha256(row,(size_t)vocab*sizeof(float),logits_hash);
            if(sd_recovery(s,&cfg,&measured,&recovery,eos,row,&scratch,err,sizeof(err)))goto done;
            ds4_bench_pool_drain(pool);pool=NULL;ds4_bench_memory_sample(getpid(),&md);
            runs++;seconds[variant]+=elapsed;block_seconds[variant]+=elapsed;
            printf("run=%d mode=mtp_same_depth repeat=%d pattern=%s slot=%d variant=%s tokens=128 cycles=%d seconds=%.9f tokens_per_second=%.6f first_cycle_ms=%.6f mean_token_ms=%.6f cycle_token_selection_exact=yes final_logits_exact=yes final_snapshot_bytes_exact=yes restored_continuation_exact=yes reference=control_mtp_same_depth correctness=integration_pending quality=pending\n",
                runs,block+1,cfg.single_run?"single":block&1?"BAAB":"ABBA",slot+1,variant?"candidate":"control",measured.cycles,elapsed,128/elapsed,1000*samples[0].seconds,1000*elapsed/128);
            printf("frontier run=%d tokens=%d checkpoint_bytes=%llu checkpoint_sha256=%s logits_sha256=%s trajectory_sha256=%s cycle_plan_sha256=%s scope=timed_final_and_checked_cycle_trajectory\n",
                runs,cfg.prefix_tokens+128,(unsigned long long)snapshot_bytes,state_hash,logits_hash,trajectory_hash,plan_hash);
            for(int c=0;c<measured.cycles;c++)printf("cycle run=%d cycle=%d offset=%d tokens=%d seconds=%.9f\n",runs,c+1,measured.frontiers[c].offset,samples[c].tokens,samples[c].seconds);
            printf("resources run=%d maxrss=%ld minor_faults=%ld major_faults=%ld input_blocks=%ld output_blocks=%ld swaps=%ld user_seconds=%.6f system_seconds=%.6f\n",
                runs,after.ru_maxrss,after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt,after.ru_inblock-before.ru_inblock,after.ru_oublock-before.ru_oublock,after.ru_nswap-before.ru_nswap,
                after.ru_utime.tv_sec-before.ru_utime.tv_sec+(after.ru_utime.tv_usec-before.ru_utime.tv_usec)/1e6,after.ru_stime.tv_sec-before.ru_stime.tv_sec+(after.ru_stime.tv_usec-before.ru_stime.tv_usec)/1e6);
            printf("memory run=%d process_valid=%d swap_valid=%d vm_valid=%d resident_bytes=%llu footprint_bytes=%llu disk_read_bytes=%llu disk_write_bytes=%llu global_swap_used_bytes=%llu global_swapins_pages=%llu global_swapouts_pages=%llu frontier=after_recovery_and_pool_drain\n",
                runs,mb.process_valid&&ma.process_valid&&md.process_valid,ma.swap_usage_valid,mb.vm_statistics_valid&&ma.vm_statistics_valid,
                (unsigned long long)md.resident_bytes,(unsigned long long)md.physical_footprint_bytes,(unsigned long long)(ma.disk_read_bytes-mb.disk_read_bytes),(unsigned long long)(ma.disk_write_bytes-mb.disk_write_bytes),
                (unsigned long long)ma.swap_used_bytes,(unsigned long long)(ma.swapins_pages-mb.swapins_pages),(unsigned long long)(ma.swapouts_pages-mb.swapouts_pages));
            fflush(stdout);
        }
        if(!cfg.single_run) {
            const double delta=log(block_seconds[0]/block_seconds[1]);paired_sum+=delta;paired_sq+=delta*delta;
            printf("paired mode=mtp_same_depth repeat=%d gain_percent=%.6f\n",block+1,100*expm1(delta));
        }
    }
    if(!cfg.single_run) {
        const double n=cfg.repeats,mean=paired_sum/n,variance=fmax(0,(paired_sq-paired_sum*mean)/(n-1));
        const double radius=student_t95(cfg.repeats-1)*sqrt(variance/n);
        printf("paired_ci mode=mtp_same_depth blocks=%d gain_percent=%.6f low95_percent=%.6f high95_percent=%.6f same_depth_equivalence=yes ordinary_equivalence=not_claimed correctness=integration_pending quality=pending accepted=no\n",cfg.repeats,100*expm1(mean),100*expm1(mean-radius),100*expm1(mean+radius));
    } else printf("single_summary mode=mtp_same_depth same_path_replay=yes candidate_evaluated=no comparison=external_pairing_required ordinary_equivalence=not_claimed correctness=integration_pending quality=pending accepted=no\n");
    for(int variant=0;variant<(cfg.single_run?1:2);variant++) {
        const int count=cfg.single_run?1:2*cfg.repeats;
        printf("aggregate mode=mtp_same_depth variant=%s runs=%d tokens=%d seconds=%.9f tokens_per_second=%.6f\n",variant?"candidate":"control",count,128*count,seconds[variant],128*count/seconds[variant]);
    }
    rc=0;
done:
    if(rc)fprintf(stderr,"mtp-same-depth: failed: %s\n",err[0]?err:"see preceding diagnostic");
    free(text);free(row);sd_free(&warm);sd_free(&measured);sd_free(&recovery);
    ds4_session_snapshot_free(&base);ds4_session_snapshot_free(&scratch);
    if(s)ds4_session_free(s);
    ds4_tokens_free(&prompt);
    if(engine)ds4_engine_close(engine);
    ds4_bench_pool_drain(pool);ds4_bench_pool_drain(outer_pool);return rc;
}
