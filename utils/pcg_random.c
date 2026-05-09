#include "pcg_random.h"
#include <omp.h> // Required for threadprivate

// Define the global variables
uint64_t pcg_state;
uint64_t pcg_inc;

// Apply threadprivate directive to the definitions
#pragma omp threadprivate(pcg_state, pcg_inc)

void pcg_seed(uint64_t initstate, uint64_t initseq)
{
    pcg_state = 0U;
    pcg_inc = (initseq << 1U) | 1U;
    (void)pcg_rand();
    pcg_state += initstate;
    (void)pcg_rand();
}

uint32_t pcg_rand(void)
{
    uint64_t oldstate = pcg_state;
    // Advance internal state
    pcg_state = oldstate * 6364136223846793005ULL + pcg_inc;
    // Calculate output function (XSH RR)
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18U) ^ oldstate) >> 27U);
    uint32_t rot = (uint32_t)(oldstate >> 59U);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t pcg_rand_bounded(uint32_t bound)
{
    // Daniel Lemire's fast unbiased bounded random method.
    // Works for any bound (including bound = 0, returns 0).
    uint64_t random32bit = pcg_rand();
    uint64_t multiresult = random32bit * bound;
    uint32_t leftover = (uint32_t)multiresult;
    if (leftover < bound)
    {
        uint32_t threshold = -bound % bound;
        while (leftover < threshold)
        {
            random32bit = pcg_rand();
            multiresult = random32bit * bound;
            leftover = (uint32_t)multiresult;
        }
    }
    return multiresult >> 32;
}

double pcg_rand_double(void)
{
    // Generate [0, 1) double using 53 bits of precision (standard double mantissa size)
    uint64_t r = ((uint64_t)pcg_rand() << 32) | pcg_rand();
    return (r >> 11) * 0x1.0p-53; // 2^-53
}