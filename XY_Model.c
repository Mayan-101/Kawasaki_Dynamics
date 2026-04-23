#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "utils/pcg_random.h"

#define WIDTH  800
#define HEIGHT 800

#define TWO_PI (2.0 * M_PI)


#define DELTA_MAX (M_PI / 4.0)

#define IDX(x, y) ((y) * WIDTH + (x))

static inline int wrap(int i, int max) { return (i + max) % max; }


static inline double wrap_angle(double a)
{
    a = fmod(a, TWO_PI);
    return (a < 0.0) ? a + TWO_PI : a;
}


static inline void hsv_to_rgb(double hue, uint8_t *r, uint8_t *g, uint8_t *b)
{
    double h6 = hue * 6.0;
    int    i  = (int)h6 % 6;
    double f  = h6 - (int)h6;
    double q  = 1.0 - f;
    switch (i) {
        case 0: *r=255;              *g=(uint8_t)(f*255); *b=0;              break;
        case 1: *r=(uint8_t)(q*255); *g=255;              *b=0;              break;
        case 2: *r=0;                *g=255;              *b=(uint8_t)(f*255); break;
        case 3: *r=0;                *g=(uint8_t)(q*255); *b=255;            break;
        case 4: *r=(uint8_t)(f*255); *g=0;                *b=255;            break;
        case 5: *r=255;              *g=0;                *b=(uint8_t)(q*255); break;
        default: *r=0; *g=0; *b=0;
    }
}


void update_grid(double *grid, int height, int width, double T, double h)
{
    for (int parity = 0; parity <= 1; parity++)
    {
#pragma omp parallel
        {
            uint64_t seed = (uint64_t)time(NULL)
                          ^ ((uint64_t)omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
            pcg_seed(seed, 42);

#pragma omp for schedule(static)
            for (int idx = 0; idx < height * width; idx++)
            {
                int x = idx % width;
                int y = idx / width;
                if (((x + y) & 1) != parity) continue;

                double theta = grid[IDX(x, y)];

                double tL = grid[IDX(wrap(x-1, width),  y               )];
                double tR = grid[IDX(wrap(x+1, width),  y               )];
                double tU = grid[IDX(x,                 wrap(y-1, height))];
                double tD = grid[IDX(x,                 wrap(y+1, height))];

                // Small random angular step
                double delta     = (pcg_rand_double() * 2.0 - 1.0) * DELTA_MAX;
                double theta_new = wrap_angle(theta + delta);

                // ΔE (J = 1)
                double dE = -(  (cos(theta_new - tL) - cos(theta - tL))
                              + (cos(theta_new - tR) - cos(theta - tR))
                              + (cos(theta_new - tU) - cos(theta - tU))
                              + (cos(theta_new - tD) - cos(theta - tD)) )
                            - h * (cos(theta_new) - cos(theta));

                // Metropolis acceptance
                if (dE <= 0.0 || (T > 0.0 && pcg_rand_double() < exp(-dE / T)))
                    grid[IDX(x, y)] = theta_new;
            }
        }
    }
}


void init_random_grid(double *grid, int height, int width)
{
    for (int i = 0; i < height * width; i++)
        grid[i] = pcg_rand_double() * TWO_PI;
}


void render_text(SDL_Renderer *renderer, TTF_Font *font, const char *text, int x, int y)
{
    if (!font) return;
    SDL_Color green   = {0, 255, 0, 255};
    SDL_Surface *surf = TTF_RenderText_Solid(font, text, green);
    if (!surf) return;
    SDL_Texture *tex  = SDL_CreateTextureFromSurface(renderer, surf);
    SDL_FreeSurface(surf);
    if (!tex) return;
    int w, th;
    SDL_QueryTexture(tex, NULL, NULL, &w, &th);
    SDL_Rect dst = {x, y, w, th};
    SDL_RenderCopy(renderer, tex, NULL, &dst);
    SDL_DestroyTexture(tex);
}


int main(int argc, char *argv[])
{
#pragma omp parallel
    {
#pragma omp single
        printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    }

    double *grid = (double *)malloc(HEIGHT * WIDTH * sizeof(double));
    if (!grid) { fprintf(stderr, "Memory allocation failed\n"); return 1; }
    init_random_grid(grid, HEIGHT, WIDTH);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        free(grid); return 1;
    }
    if (TTF_Init() < 0)
        fprintf(stderr, "TTF_Init Error: %s\n", TTF_GetError());

    SDL_Window *window = SDL_CreateWindow("XY Model — continuous spins",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) { SDL_Quit(); free(grid); return 1; }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) { SDL_DestroyWindow(window); SDL_Quit(); free(grid); return 1; }

    TTF_Font *font = TTF_OpenFont("C:/Windows/Fonts/consola.ttf", 24);
    if (!font) fprintf(stderr, "Font loading failed: %s\n", TTF_GetError());

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_RGBA32,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             WIDTH, HEIGHT);
    if (!texture) {
        SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window);
        TTF_CloseFont(font); TTF_Quit(); SDL_Quit(); free(grid); return 1;
    }

    uint32_t *pixels = (uint32_t *)malloc(WIDTH * HEIGHT * sizeof(uint32_t));

    
    double T = 0.89;
    double h = 0.0;

    const double T_MIN = 0.0,  T_MAX = 5.0, T_STEP = 0.05;
    const double H_MIN = -5.0, H_MAX = 5.0, H_STEP = 0.1;

    int running = 1;
    SDL_Event event;

    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT) { running = 0; }
            else if (event.type == SDL_KEYDOWN)
            {
                switch (event.key.keysym.sym)
                {
                case SDLK_UP:    T += T_STEP; if (T > T_MAX) T = T_MAX; break;
                case SDLK_DOWN:  T -= T_STEP; if (T < T_MIN) T = T_MIN; break;
                case SDLK_RIGHT: h += H_STEP; if (h > H_MAX) h = H_MAX; break;
                case SDLK_LEFT:  h -= H_STEP; if (h < H_MIN) h = H_MIN; break;
                }
            }
        }

        update_grid(grid, HEIGHT, WIDTH, T, h);

        // Render: map θ → hue  (red = 0, yellow = π/3, green = 2π/3, …)
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                double hue = grid[IDX(x, y)] / TWO_PI;   // [0, 1)
                uint8_t r, g, b;
                hsv_to_rgb(hue, &r, &g, &b);
                pixels[y * WIDTH + x] = (255u << 24) | (b << 16) | (g << 8) | r;
            }
        }

        SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // HUD
        char buf[48];
        snprintf(buf, sizeof(buf), "T = %.2f  [Up/Down]",     T);
        render_text(renderer, font, buf, 10, 10);
        snprintf(buf, sizeof(buf), "B = %+.2f  [Left/Right]", h);
        render_text(renderer, font, buf, 10, 40);

        SDL_RenderPresent(renderer);
        SDL_Delay(10);
    }

    free(pixels);
    free(grid);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_Quit();
    return 0;
}