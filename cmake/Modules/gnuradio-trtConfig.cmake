find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_TRT gnuradio-trt)

FIND_PATH(
    GR_TRT_INCLUDE_DIRS
    NAMES gnuradio/trt/api.h
    HINTS $ENV{TRT_DIR}/include
        ${PC_TRT_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_TRT_LIBRARIES
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

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-trtTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_TRT DEFAULT_MSG GR_TRT_LIBRARIES GR_TRT_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_TRT_LIBRARIES GR_TRT_INCLUDE_DIRS)
