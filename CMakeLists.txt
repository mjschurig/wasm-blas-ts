cmake_minimum_required(VERSION 3.15)
project(wasm-blas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    src/cpp/daxpy.cpp
    src/cpp/dcopy.cpp
    src/cpp/ddot.cpp
    src/cpp/dscal.cpp
    src/cpp/dasum.cpp
    src/cpp/dnrm2.cpp
    src/cpp/dswap.cpp
    src/cpp/drot.cpp
    src/cpp/drotg.cpp
    src/cpp/drotm.cpp
    src/cpp/daxpby.cpp
    src/cpp/drotmg.cpp
    src/cpp/dgemv.cpp
    src/cpp/dger.cpp
    src/cpp/dsymv.cpp
    src/cpp/dsyr.cpp
    src/cpp/dgemm.cpp
    src/cpp/dsymm.cpp
    src/cpp/dsyrk.cpp
    src/cpp/dsyr2.cpp
    src/cpp/dtrmv.cpp
    src/cpp/dtrsv.cpp
    src/cpp/dsyr2k.cpp
    src/cpp/dtrmm.cpp
    src/cpp/dtrsm.cpp
    src/cpp/dgbmv.cpp
    src/cpp/dsbmv.cpp
    src/cpp/dspmv.cpp
    src/cpp/dspr.cpp
    src/cpp/dspr2.cpp
    src/cpp/dtbmv.cpp
    src/cpp/dtbsv.cpp
    src/cpp/dtpmv.cpp
    src/cpp/dtpsv.cpp
    src/cpp/dgemmtr.cpp
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
        "SHELL:-s EXPORTED_FUNCTIONS=['_daxpy','_dcopy','_ddot','_dscal','_dasum','_dnrm2','_dswap','_drot','_drotg','_drotm','_daxpby','_drotmg','_dgemv','_dger','_dsymv','_dsyr','_dsyr2','_dtrmv','_dtrsv','_dgemm','_dsymm','_dsyrk','_dsyr2k','_dtrmm','_dtrsm','_dgbmv','_dsbmv','_dspmv','_dspr','_dspr2','_dtbmv','_dtbsv','_dtpmv','_dtpsv','_dgemmtr','_malloc','_free']"
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
