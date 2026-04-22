cmake_minimum_required(VERSION 3.10)
project(Kawasaki_Dynamics C)

set(CMAKE_C_STANDARD 11)

# Find SDL2 using pkg-config (reliable for MSYS2)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

find_package(SDL2_ttf REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(Kawasaki_Dynamics
    Vapour_model.c
    utils/pcg_random.c
)

target_include_directories(Kawasaki_Dynamics PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(Kawasaki_Dynamics ${SDL2_LIBRARIES} SDL2_ttf::SDL2_ttf m OpenMP::OpenMP_C)

# Ensure console window for printf debugging (MinGW)
if(MINGW)
    set_target_properties(Kawasaki_Dynamics PROPERTIES
        LINK_FLAGS "-mconsole"
    )
endif()