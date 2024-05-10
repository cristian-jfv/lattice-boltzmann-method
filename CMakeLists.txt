cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(LatticeBoltzmannMethod
	VERSION 1.0)

if(DEFINED ENV{Torch_DIR})
message(STATUS "torch installation prefic=$ENV{Torch_DIR}")
else()
message(FATAL_ERROR "Torch_DIR not defined")
endif()

if(DEFINED ENV{TOMLCPP_DIR})
message(STATUS "tomlcplusplus include directory=$ENV{TOMLCPP_DIR}")
else()
message(FATAL_ERROR "tomlcplusplus not defined")
endif()

include_directories("$ENV{TOMLCPP_DIR}")

find_package(Torch REQUIRED)

add_compile_options(
       -Wall -Werror -Wno-error=maybe-uninitialized
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)

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

add_library(ibm OBJECT src/ibm.cpp)
set_property(TARGET ibm PROPERTY CXX_STANDARD 17)
target_link_libraries(ibm "${TORCH_LIBRARIES}")
target_include_directories(ibm ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(utils OBJECT src/utils.cpp)
set_property(TARGET utils PROPERTY CXX_STANDARD 17)
target_link_libraries(utils "${TORCH_LIBRARIES}")
target_include_directories(utils ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

# Executables
add_executable(domain_boundaries_test test/domain_boundaries_test.cpp)
set_property(TARGET domain_boundaries_test PROPERTY CXX_STANDARD 17)
target_link_libraries(domain_boundaries_test
		PRIVATE "${TORCH_LIBRARIES}" params solver domain)

add_executable(horizontal_poiseuille_test test/horizontal_poiseuille_test.cpp)
set_property(TARGET horizontal_poiseuille_test PROPERTY CXX_STANDARD 17)
target_link_libraries(horizontal_poiseuille_test
		PRIVATE "${TORCH_LIBRARIES}" params solver domain utils)

add_executable(vertical_poiseuille_test test/vertical_poiseuille_test.cpp)
set_property(TARGET vertical_poiseuille_test PROPERTY CXX_STANDARD 17)
target_link_libraries(vertical_poiseuille_test
		PRIVATE "${TORCH_LIBRARIES}" params solver domain utils)

add_executable(specular_boundary_test test/specular_boundary_test.cpp)
set_property(TARGET specular_boundary_test PROPERTY CXX_STANDARD 17)
target_link_libraries(specular_boundary_test
		PRIVATE "${TORCH_LIBRARIES}" params solver domain utils)

add_executable(gravity_test test/gravity_test.cpp)
set_property(TARGET gravity_test PROPERTY CXX_STANDARD 17)
target_link_libraries(gravity_test
		PRIVATE "${TORCH_LIBRARIES}" solver utils)

add_executable(cylinder_test test/cylinder_test.cpp)
set_property(TARGET cylinder_test PROPERTY CXX_STANDARD 17)
target_link_libraries(cylinder_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils ibm)

