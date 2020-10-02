INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_TRT trt)

FIND_PATH(
    TRT_INCLUDE_DIRS
    NAMES trt/api.h
    HINTS $ENV{TRT_DIR}/include
        ${PC_TRT_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    TRT_LIBRARIES
    NAMES gnuradio-trt
    HINTS $ENV{TRT_DIR}/lib
        ${PC_TRT_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/trtTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TRT DEFAULT_MSG TRT_LIBRARIES TRT_INCLUDE_DIRS)
MARK_AS_ADVANCED(TRT_LIBRARIES TRT_INCLUDE_DIRS)
