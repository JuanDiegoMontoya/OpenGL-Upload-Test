# the project's main CMakeLists file

cmake_minimum_required(VERSION 3.14)

project(tests)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external)

set(source_files
	src/main.cpp
	src/stb_image.h
)

add_executable(engine ${source_files})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT engine)
target_include_directories(engine PUBLIC src)

find_package(OpenGL REQUIRED)

# enable asan for debug builds
if (DEBUG)
    if (WIN32)
        target_compile_options(engine PUBLIC /fsanitize=address)
    else()
        target_compile_options(engine PUBLIC -fsanitize=address)
    endif()
endif()

# Determine whether we're compiling with clang++
string(FIND "${CMAKE_CXX_COMPILER}" "clang++" GAME_COMPILER_CLANGPP)
if(GAME_COMPILER_CLANGPP GREATER -1)
  set(GAME_COMPILER_CLANGPP 1)
else()
  set(GAME_COMPILER_CLANGPP 0)
endif()

target_compile_options(engine
	INTERFACE
	$<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,${GAME_COMPILER_CLANGPP}>:
	-Wall
	-Wextra
	-Wconversion
	-Wsign-conversion>
	$<$<CXX_COMPILER_ID:MSVC>:
	/W3
	/std=c++latest # this doesn't seem to actually do anything
	>
)

# copies assets to the build folder
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets)
add_dependencies(engine copy_assets)

target_link_libraries(engine glfw lib_glad)