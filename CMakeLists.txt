# cmake_minimum_required(VERSION 3.18)
# project(MLP LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CUDA_ARCHITECTURES 120)

# find_package(CUDAToolkit REQUIRED)

# file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
#     src/*.cpp
# )

# file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
#     src/*.cu
# )

# add_executable(mlp
#     ${CPP_SOURCES}
#     ${CUDA_SOURCES}
# )

# target_include_directories(mlp PRIVATE
#     src
#     ${CUDAToolkit_INCLUDE_DIRS}
# )

# target_link_libraries(mlp PRIVATE CUDA::cudart)

cmake_minimum_required(VERSION 3.18)

project(MLP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MLP_COMPUTE "HOST" CACHE STRING "MLP compute context: CUDA, OPENCL, or leave blank for host computing")
set_property(CACHE MLP_COMPUTE PROPERTY STRINGS CUDA OPENCL HOST)

option(MLP_DEBUG "Enable debug mode" OFF)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

list(FILTER CPP_SOURCES EXCLUDE REGEX "/(cuda|host|opencl)/")

set(SOURCES ${CPP_SOURCES})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")

if(MLP_COMPUTE STREQUAL "CUDA")

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_CUDA_ARCHITECTURES 120)

    find_package(CUDAToolkit REQUIRED)

    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        src/cuda/*.cu
        src/cuda/*.cpp
    )

    list(APPEND SOURCES ${CUDA_SOURCES})

    add_executable(mlp ${SOURCES})

    target_compile_definitions(mlp PRIVATE MLP_CUDA)

    target_include_directories(mlp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    target_link_libraries(mlp PRIVATE CUDA::cudart)

elseif(MLP_COMPUTE STREQUAL "OPENCL")

    find_package(OpenCL REQUIRED)

    file(GLOB_RECURSE OPENCL_SOURCES CONFIGURE_DEPENDS
        src/opencl/*.cpp
    )

    list(APPEND SOURCES ${OPENCL_SOURCES})

    add_executable(mlp ${SOURCES})

    target_compile_definitions(mlp PRIVATE MLP_OPENCL)

    target_link_libraries(mlp PRIVATE OpenCL::OpenCL)

elseif(MLP_COMPUTE STREQUAL "HOST")

    file(GLOB_RECURSE HOST_SOURCES CONFIGURE_DEPENDS
        src/host/*.cpp
    )

    list(APPEND SOURCES ${HOST_SOURCES})

    add_executable(mlp ${SOURCES})

    target_compile_definitions(mlp PRIVATE MLP_HOST)

else()

    message(FATAL_ERROR
        "Unrecognised compute context: '${MLP_COMPUTE}'"
        "Choose from CUDA, OPENCL, or HOST"
    )

endif()

target_compile_definitions(mlp PRIVATE RANDOM_BS)

target_include_directories(mlp PRIVATE src)