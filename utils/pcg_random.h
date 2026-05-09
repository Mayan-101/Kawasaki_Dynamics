#ifndef PCG_RANDOM_H
#define PCG_RANDOM_H

#include <stdint.h>

extern uint64_t pcg_state;
extern uint64_t pcg_inc;

void pcg_seed(uint64_t initstate, uint64_t initseq);
uint32_t pcg_rand(void);
uint32_t pcg_rand_bounded(uint32_t bound);
double pcg_rand_double(void);

#endif