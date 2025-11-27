cmake_minimum_required(VERSION 3.12.0)
project(lab02 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_library(Poisson1D STATIC src/Poisson1D.cpp)
deal_ii_setup_target(Poisson1D)

add_executable(exercise-01 src/exercise-01.cpp)
target_link_libraries(exercise-01 Poisson1D)
deal_ii_setup_target(exercise-01)

add_executable(exercise-02 src/exercise-02.cpp)
target_link_libraries(exercise-02 Poisson1D)
deal_ii_setup_target(exercise-02)
