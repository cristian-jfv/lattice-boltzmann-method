cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

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

#add_library(rk OBJECT src/rk.cpp)
#set_property(TARGET rk PROPERTY CXX_STANDARD 17)
#target_link_libraries(rk "${TORCH_LIBRARIES}")
#target_include_directories(rk ${warning_guard} PUBLIC
#				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(colour OBJECT src/colour.cpp)
set_property(TARGET colour PROPERTY CXX_STANDARD 17)
target_link_libraries(colour "${TORCH_LIBRARIES}")
target_include_directories(colour ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(differential OBJECT src/differential.cpp)
set_property(TARGET differential PROPERTY CXX_STANDARD 17)
target_link_libraries(differential "${TORCH_LIBRARIES}")
target_include_directories(differential ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

add_library(ulbm OBJECT src/ulbm.cpp)
set_property(TARGET ulbm PROPERTY CXX_STANDARD 17)
target_link_libraries(ulbm "${TORCH_LIBRARIES}")
target_include_directories(ulbm ${warning_guard} PUBLIC
				"\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src")

# Executables

add_executable(horizontal_poiseuille_test test/horizontal_poiseuille_test.cpp)
set_property(TARGET horizontal_poiseuille_test PROPERTY CXX_STANDARD 17)
target_link_libraries(horizontal_poiseuille_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils)

add_executable(vertical_poiseuille_test test/vertical_poiseuille_test.cpp)
set_property(TARGET vertical_poiseuille_test PROPERTY CXX_STANDARD 17)
target_link_libraries(vertical_poiseuille_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils)

add_executable(specular_boundary_test test/specular_boundary_test.cpp)
set_property(TARGET specular_boundary_test PROPERTY CXX_STANDARD 17)
target_link_libraries(specular_boundary_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils)

add_executable(gravity_test test/gravity_test.cpp)
set_property(TARGET gravity_test PROPERTY CXX_STANDARD 17)
target_link_libraries(gravity_test
		PRIVATE "${TORCH_LIBRARIES}" solver utils)

add_executable(cylinder_test test/cylinder_test.cpp)
set_property(TARGET cylinder_test PROPERTY CXX_STANDARD 17)
target_link_libraries(cylinder_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils ibm)

add_executable(rectangle_sedimentation_test test/rectangle_sedimentation_test.cpp)
set_property(TARGET rectangle_sedimentation_test PROPERTY CXX_STANDARD 17)
target_link_libraries(rectangle_sedimentation_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils ibm)

add_executable(free_stream_test test/free_stream_test.cpp)
set_property(TARGET free_stream_test PROPERTY CXX_STANDARD 17)
target_link_libraries(free_stream_test
		PRIVATE "${TORCH_LIBRARIES}" params solver utils ibm)

add_executable(static_droplet_test test/static_droplet_test.cpp)
set_property(TARGET static_droplet_test PROPERTY CXX_STANDARD 17)
target_link_libraries(static_droplet_test
		PRIVATE "${TORCH_LIBRARIES}" solver utils)

add_executable(rk_static_droplet_test test/rk_static_droplet_test.cpp)
set_property(TARGET rk_static_droplet_test PROPERTY CXX_STANDARD 17)
target_link_libraries(rk_static_droplet_test
		PRIVATE "${TORCH_LIBRARIES}" solver utils)

#add_executable(rk_droplet_test test/rk_droplet_test.cpp)
#set_property(TARGET rk_droplet_test PROPERTY CXX_STANDARD 17)
#target_link_libraries(rk_droplet_test
#		PRIVATE "${TORCH_LIBRARIES}" solver utils rk)
#
#add_executable(rk_rayleigh_taylor_test test/rk_rayleigh_taylor_test.cpp)
#set_property(TARGET rk_rayleigh_taylor_test PROPERTY CXX_STANDARD 17)
#target_link_libraries(rk_rayleigh_taylor_test
#		PRIVATE "${TORCH_LIBRARIES}" solver utils rk)

add_executable(mrtcg_static_droplet test/mrtcg_static_droplet.cpp)
set_property(TARGET mrtcg_static_droplet PROPERTY CXX_STANDARD 17)
target_link_libraries(mrtcg_static_droplet
		PRIVATE "${TORCH_LIBRARIES}" solver utils colour differential)

add_executable(mrtcg_rayleigh_taylor test/mrtcg_rayleigh_taylor.cpp)
set_property(TARGET mrtcg_rayleigh_taylor PROPERTY CXX_STANDARD 17)
target_link_libraries(mrtcg_rayleigh_taylor
		PRIVATE "${TORCH_LIBRARIES}" solver utils colour differential)

add_executable(mrtcg_multiple_mode_rayleigh_taylor test/mrtcg_multiple_mode_rayleigh_taylor.cpp)
set_property(TARGET mrtcg_multiple_mode_rayleigh_taylor PROPERTY CXX_STANDARD 17)
target_link_libraries(mrtcg_multiple_mode_rayleigh_taylor
		PRIVATE "${TORCH_LIBRARIES}" solver utils colour differential)

add_executable(mrt_rayleigh_taylor test/mrt_rayleigh_taylor.cpp)
set_property(TARGET mrt_rayleigh_taylor PROPERTY CXX_STANDARD 17)
target_link_libraries(mrt_rayleigh_taylor
		PRIVATE "${TORCH_LIBRARIES}" solver utils colour differential)

add_executable(decompose_domain test/decompose_domain.cpp)
set_property(TARGET decompose_domain PROPERTY CXX_STANDARD 17)
target_link_libraries(decompose_domain
		PRIVATE "${TORCH_LIBRARIES}" solver utils domain)

add_executable(decompose_domain_loop test/decompose_domain_loop.cpp)
set_property(TARGET decompose_domain_loop PROPERTY CXX_STANDARD 17)
target_link_libraries(decompose_domain_loop
		PRIVATE "${TORCH_LIBRARIES}" solver utils domain)

add_executable(ulbm_poiseuille test/ulbm_poiseuille.cpp)
set_property(TARGET ulbm_poiseuille PROPERTY CXX_STANDARD 17)
target_link_libraries(ulbm_poiseuille
		PRIVATE "${TORCH_LIBRARIES}" solver utils ulbm)

add_executable(ulbm_double_shear_flow test/ulbm_double_shear_flow.cpp)
set_property(TARGET ulbm_double_shear_flow PROPERTY CXX_STANDARD 17)
target_link_libraries(ulbm_double_shear_flow
		PRIVATE "${TORCH_LIBRARIES}" solver utils ulbm)
