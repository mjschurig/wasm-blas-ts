cmake_minimum_required(VERSION 3.15)
project(wasm-blas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    src/cpp/daxpy.cpp
)

# Create the WebAssembly library
add_executable(blas ${SOURCES})

# Emscripten-specific settings
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    
    # Emscripten compile and link flags
    set(EMSCRIPTEN_COMPILE_FLAGS
        -O3
    )
    
    set(EMSCRIPTEN_LINK_FLAGS
        -O3
        "SHELL:-s WASM=1"
        "SHELL:-s EXPORTED_FUNCTIONS=['_daxpy','_malloc','_free']"
        "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF64','HEAP8','HEAPU8']"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s EXPORT_NAME='createBlasModule'"
        "SHELL:-s ENVIRONMENT='web,worker,node'"
        --no-entry
    )
    
    target_compile_options(blas PRIVATE ${EMSCRIPTEN_COMPILE_FLAGS})
    target_link_options(blas PRIVATE ${EMSCRIPTEN_LINK_FLAGS})
endif()
