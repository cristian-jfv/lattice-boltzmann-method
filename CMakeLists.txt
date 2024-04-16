cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(LatticeBoltzmannMethod
	VERSION 1.0)

find_package(Torch REQUIRED)

# Libraries
add_library(params OBJECT src/params.cpp)
target_include_directories(params ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(solver OBJECT src/solver.cpp)
set_property(TARGET solver PROPERTY CXX_STANDARD 17)
target_link_libraries(solver "${TORCH_LIBRARIES}")
target_include_directories(solver ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(domain OBJECT src/domain.cpp)
set_property(TARGET domain PROPERTY CXX_STANDARD 17)
target_link_libraries(domain "${TORCH_LIBRARIES}")
target_include_directories(domain ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_executable(lbm src/main.cpp)
set_property(TARGET lbm PROPERTY CXX_STANDARD 17)
target_link_libraries(lbm PRIVATE "${TORCH_LIBRARIES}" params solver domain)

add_executable(test test/test.cpp)
set_property(TARGET test PROPERTY CXX_STANDARD 17)
target_link_libraries(test PRIVATE "${TORCH_LIBRARIES}" params solver domain)
