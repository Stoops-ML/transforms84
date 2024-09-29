cmake_minimum_required(VERSION 3.14...3.30)

project(
    Transforms84
    VERSION 1.0
    LANGUAGES C
)

add_library(Transforms84
    include/definitions.h
    include/general.h
    include/distances.c
    include/transforms.c
    include/helpers.c
)

find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

# print found python path

target_link_libraries(Transforms84 PRIVATE Python3::Python)
target_link_libraries(Transforms84 PRIVATE Python3::NumPy)
target_link_libraries(Transforms84 PRIVATE OpenMP::OpenMP_C)
target_include_directories(Transforms84 PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(Transforms84 PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

target_include_directories(Transforms84 PUBLIC include)

add_executable(Transforms84_Tester
    tests/test.c
)


target_link_libraries(Transforms84_Tester PRIVATE Transforms84)
target_include_directories(Transforms84_Tester PRIVATE include)
target_link_libraries(Transforms84_Tester PRIVATE Python3::Python)
target_link_libraries(Transforms84_Tester PRIVATE Python3::NumPy)
target_link_libraries(Transforms84_Tester PRIVATE OpenMP::OpenMP_C)
target_include_directories(Transforms84_Tester PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(Transforms84_Tester PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories(Transforms84_Tester PRIVATE ${OpenMP_C_INCLUDE_DIRS})

