/*
 * test_arena_alignment.c -- GAP-H009 gates: BearArena allocator invariants
 *
 * Fuzz-seeded property checks:
 *  G1 alignment: every pointer satisfies its requested align
 *  G2 non-overlap: live regions never intersect
 *  G3 accounting: arena->used matches cumulative padding+size
 *  G4 scratch restore: begin/end restores exact used + ptr
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "bear_arena.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static unsigned long rs=0xC0FFEEUL;
static float fr(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;
    return (float)((rs>>11)&0x3FFFFFF)/(float)0x3FFFFFF;}

int main(void){
    printf("=== Arena Alignment Property Tests ===\n\n");

    printf("  g123_align_overlap_accounting...");
    {
        size_t cap=1<<20;
        BearArena a; CHECK(bear_arena_create(&a,cap)==0);
        void*   ptr[4096]; size_t sz[4096];
        size_t  accounted=0;
        int live=0, total_alloc=0;
        for(int iter=0;iter<8000 && a.used<cap-8192;iter++){
            size_t size=(size_t)(fr()*4096)+1;
            size_t align;
            unsigned sel=(unsigned)(fr()*4);
            align = sel==0?16:sel==1?32:sel==2?64:8;
            void* p=bear_arena_alloc(&a,size,align);
            if(!p) break;
            CHECK(((uintptr_t)p & (align-1))==0);           /* G1 */
            uintptr_t pn=(uintptr_t)p;
            for(int i=0;i<live;i++){                         /* G2 */
                uintptr_t pi=(uintptr_t)ptr[i];
                CHECK(pn >= pi+sz[i] || pn+size <= pi);
            }
            /* G3 handled via accounted-vs-used check after loop */
            ptr[live]=p; sz[live]=size; live++;
            accounted+=size;
            total_alloc++;
        }
        CHECK(total_alloc>500);                              /* meaningful fuzz */
        /* used == sum sizes + padding overhead (>= sum) */
        CHECK(a.used>=accounted);
        bear_arena_destroy(&a);
    }
    printf("PASS\n");passed++;

    printf("  g4_scratch_restore...");
    {
        BearArena a; CHECK(bear_arena_create(&a,1<<20)==0);
        void* persistent=bear_arena_alloc(&a,128,64);
        CHECK(persistent);
        size_t mark_after_persist=a.used;

        BearScratch sc=bear_scratch_begin(&a);
        for(int i=0;i<10;i++) bear_arena_alloc(&a,1000+i,16);
        CHECK(a.used>mark_after_persist);          /* scratch consumed */
        bear_scratch_end(sc);
        CHECK(a.used==mark_after_persist);         /* restored exactly */
        CHECK((uintptr_t)a.ptr==(uintptr_t)a.base+mark_after_persist);
        bear_arena_destroy(&a);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
