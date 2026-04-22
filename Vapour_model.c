#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "utils/pcg_random.h"

#define WIDTH 600
#define HEIGHT 600

// Convert 2D coordinates to 1D index
#define IDX(x, y) ((y) * WIDTH + (x))

// Periodic boundary helper
static inline int wrap(int i, int max)
{
    return (i + max) % max;
}

void update_grid(int *grid, int height, int width, double T)
{
#pragma omp parallel
    {
        uint64_t seed = (uint64_t)time(NULL) ^ (omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
        pcg_seed(seed, 42);
#pragma omp for
        for (int idx = 0; idx < height * width; idx++)
        {
            int x = idx % width;
            int y = idx / width;
            if (((x + y) & 1) != 0)
                continue;
            int current = grid[IDX(x, y)];

            int left = grid[IDX(wrap(x - 1, width), y)];
            int right = grid[IDX(wrap(x + 1, width), y)];
            int up = grid[IDX(x, wrap(y - 1, height))];
            int down = grid[IDX(x, wrap(y + 1, height))];

            int neighbor_sum = left + right + up + down;
            int delta_E = 2 * current * neighbor_sum;

            if (delta_E <= 0 || pcg_rand_double() < exp(-delta_E / T))
            {
                grid[IDX(x, y)] = -current;
            }
        }
    }
#pragma omp parallel
    {
        uint64_t seed = (uint64_t)time(NULL) ^ (omp_get_thread_num() * 0x9E3779B97F4A7C15ULL);
        pcg_seed(seed, 42);
#pragma omp for
        for (int idx = 0; idx < height * width; idx++)
        {
            int x = idx % width;
            int y = idx / width;
            if (((x + y) & 1) != 1)
                continue;
            int current = grid[IDX(x, y)];

            int left = grid[IDX(wrap(x - 1, width), y)];
            int right = grid[IDX(wrap(x + 1, width), y)];
            int up = grid[IDX(x, wrap(y - 1, height))];
            int down = grid[IDX(x, wrap(y + 1, height))];

            int neighbor_sum = left + right + up + down;
            int delta_E = 2 * current * neighbor_sum;

            if (delta_E <= 0 || pcg_rand_double() < exp(-delta_E / T))
            {
                grid[IDX(x, y)] = -current;
            }
        }
    }
}

void init_random_grid(int *grid, int height, int width)
{
    for (int i = 0; i < height * width; i++)
    {
        grid[i] = (pcg_rand_bounded(1) == 0) ? -1 : 1;
    }
}

// Render text on screen
void render_text(SDL_Renderer *renderer, TTF_Font *font, const char *text, int x, int y)
{
    if (!font)
        return;
    SDL_Color white = {255, 0, 0, 255};
    SDL_Surface *surface = TTF_RenderText_Solid(font, text, white);
    if (!surface)
        return;
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    if (!texture)
        return;

    int texW = 0, texH = 0;
    SDL_QueryTexture(texture, NULL, NULL, &texW, &texH);
    SDL_Rect dst = {x, y, texW, texH};
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

int main(int argc, char *argv[])
{
#pragma omp parallel
    {
#pragma omp single
        printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    }
    // Allocate grid
    int *grid = (int *)malloc(HEIGHT * WIDTH * sizeof(int));
    if (!grid)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    init_random_grid(grid, HEIGHT, WIDTH);

    // SDL Initialization
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        free(grid);
        return 1;
    }

    // Initialize SDL_ttf
    if (TTF_Init() < 0)
    {
        fprintf(stderr, "TTF_Init Error: %s\n", TTF_GetError());
    }

    SDL_Window *window = SDL_CreateWindow("Vapour Model",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          WIDTH, HEIGHT,
                                          SDL_WINDOW_SHOWN);
    if (!window)
    {
        SDL_Quit();
        free(grid);
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1,
                                                SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        SDL_DestroyWindow(window);
        SDL_Quit();
        free(grid);
        return 1;
    }

    TTF_Font *font = TTF_OpenFont("C:/Windows/Fonts/consola.ttf", 24);
    if (!font)
    {
        fprintf(stderr, "Font loading failed: %s\n", TTF_GetError());
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_RGBA32,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             WIDTH, HEIGHT);
    if (!texture)
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_CloseFont(font);
        TTF_Quit();
        SDL_Quit();
        free(grid);
        return 1;
    }

    // Main loop
    int running = 1;
    SDL_Event event;
    uint32_t *pixel_buffer = (uint32_t *)malloc(WIDTH * HEIGHT * sizeof(uint32_t));

    // Temperature variable
    double T = 1.5;
    const double T_MIN = 0.0;
    const double T_MAX = 5.0;
    const double T_STEP = 0.1;

    while (running)
    {
        // Handle events
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = 0;
            }
            else if (event.type == SDL_KEYDOWN)
            {
                switch (event.key.keysym.sym)
                {
                case SDLK_UP:
                    T += T_STEP;
                    if (T > T_MAX)
                        T = T_MAX;
                    break;
                case SDLK_DOWN:
                    T -= T_STEP;
                    if (T < T_MIN)
                        T = T_MIN;
                    break;
                }
            }
        }

        // Monte Carlo sweep with current temperature
        update_grid(grid, HEIGHT, WIDTH, T);

        // Convert liq_phases to pixels
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                int liq_phase = grid[IDX(x, y)];
                uint8_t r, g, b;
                if (liq_phase == 1)
                {
                    r = 255;
                    g = 255;
                    b = 255;
                }
                else
                {
                    r = 073;
                    g = 016;
                    b = 230;
                }

                uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
                pixel_buffer[y * WIDTH + x] = pixel;
            }
        }

        // Render everything
        SDL_UpdateTexture(texture, NULL, pixel_buffer, WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Display temperature in top‑left corner
        char temp_text[32];
        snprintf(temp_text, sizeof(temp_text), "T = %.2f", T);
        render_text(renderer, font, temp_text, 10, 10);

        SDL_RenderPresent(renderer);
        SDL_Delay(10);
    }

    free(pixel_buffer);
    free(grid);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_Quit();

    return 0;
}