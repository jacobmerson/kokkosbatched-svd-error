project(test-svd)
cmake_minimum_required(VERSION 3.20)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

add_executable(test test.cc)
target_link_libraries(test PUBLIC Kokkos::kokkoskernels)
