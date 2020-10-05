# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT trt_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT trt_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################
list(APPEND trt_python_files
    infer_python.cc
    fft_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(trt 
   ../..
   gr::trt
   "${trt_python_files}")

install(TARGETS trt_python DESTINATION ${GR_PYTHON_DIR}/trt COMPONENT pythonapi)
