cmake_minimum_required(VERSION 3.10)
project(Kawasaki_Dynamics C)

set(CMAKE_C_STANDARD 11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)
find_package(SDL2_ttf REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(xy_interactive
    XY_model_gpu.c
    gpu_loader.c
    utils/pcg_random.c
)

target_include_directories(xy_interactive PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(xy_interactive
    ${SDL2_LIBRARIES}
    SDL2_ttf::SDL2_ttf
    m
    OpenMP::OpenMP_C
)

if(MINGW)
    set_target_properties(xy_interactive PROPERTIES LINK_FLAGS "-mconsole")
endif()