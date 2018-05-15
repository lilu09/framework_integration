TARGET_EXT=cu

#---------------------------------------------------------------------#
TYPE=bin
#TARGET=compose
#---------------------------------------------------------------------#
#TYPE=static
#LIB_NAME=hello
#LIB_HEADER=x.h
#LIB_HEADER_INSTALL_ROOT_DIR=/home/lu/Downloads
#LIB_BIN_INSTALL_ROOT_DIR=/home/lu/Downloads
#---------------------------------------------------------------------#

#remember that no space after MACHINE_NAME or MODULE_LOAD
MACHINE_NAME=M4800#EXCESS
MODULE_LOAD=BLAS OPENMP# CLANG NVML PCM CUPTI OPENCL OPENGL PTHREAD CXX_PTHREAD R RCPP RINSIDE
NUM_OF_VIRTUAL_CORES:=$(shell cat /proc/cpuinfo | grep "^processor" | wc -l)
#make cool EC=-DDEBUG #make allclean cool EC="-DVERBOSE=1"

#USR_INC_SEARCH_PATH=x y

EC=-D_FORCE_INLINES -DXPDL_NUM_OF_GPUS=1 -DXPDL_NUM_OF_HW_THREADS=$(NUM_OF_VIRTUAL_CORES) -DENABLE_CUDA_TIME 
EL=-lcublas
#EC=-g -DDEBUG #Extra Compile flag
#EL=-g -DDEBUG #Extra Linker flag
#RP=sudo #Running Prefix
#EF=$(PCM_SRC)/cpucounters.cpp #Extra Files
#EXCLUDE_SRC_FILES=funcallsample.cpp
#ERP:=1 2

#DEBUG=BATCH
#DEBUG=INTERACTIVE
#DEBUG=BACK_TRACE


NO_CUDA_MEMCHECK:=true

#PROFILE=true

#NO_STRIP=true

NO_CPPCHECK=true

#NO_WARNINGS:=true

#NO_OPTIMIZATION:=true

#ENABLE_RTTI:=true


#If App generate some file, put it here, allclean target wil delete it.
#APP_FILES=idlist
#Separete app files, such that you can view the source and execute the binary
#APP_FILE1=idlist
#APP_FILE2=idlist

#If you want to see the app generated files, put it here, it will be executed after app runs.
#APP_VIEW_RESULT=vi $(APP_FILES)

include MakefileCommon
