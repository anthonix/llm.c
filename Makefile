CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Find nvcc
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# Default O3 CPU optimization level for NVCC (0 for fastest compile time)
FORCE_NVCC_O ?= 3

# NVCC flags
# -t=0 is short for --threads, 0 = number of CPUs on the machine
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_LDFLAGS = -lcublas -lcublasLt
NVCC_INCLUDES =
NVCC_LDLIBS =
NCLL_INCUDES =
NVCC_CUDNN =
# By default we don't build with cudnn because it blows up compile time from a few seconds to ~minute
USE_CUDNN ?= 0

# We will place .o files in the `build` directory (create it if it doesn't exist)
BUILD_DIR = build
ifeq ($(OS), Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
endif

# Function to check if a file exists in the PATH
ifneq ($(OS), Windows_NT)
define file_exists_in_path
  $(which $(1) 2>/dev/null)
endef
else
define file_exists_in_path
  $(shell where $(1) 2>nul)
endef
endif

ifneq ($(CI),true) # if not in CI, then use the GPU query
  ifndef GPU_COMPUTE_CAPABILITY # set to defaults if: make GPU_COMPUTE_CAPABILITY=
    ifneq ($(call file_exists_in_path, nvidia-smi),)
      # Get the compute capabilities of all GPUs
      # Remove decimal points, sort numerically in ascending order, and select the first (lowest) value
      GPU_COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
      GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
    endif
  endif
endif

# set to defaults if - make GPU_COMPUTE_CAPABILITY= otherwise use the compute capability detected above
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# AMD flags
ROCM_PATH ?= /opt/rocm
AMDGPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
HIPCC := $(shell which hipcc 2>/dev/null)
HIPIFY := $(shell which hipify-perl 2>/dev/null)
HIPCC_FLAGS = -O3 -march=native -I$(BUILD_DIR)/hip -fno-strict-aliasing
HIPCC_LDFLAGS += -lamdhip64 -lhipblaslt
ifneq ($(filter gfx1100,$(AMDGPU_TARGETS)),)
  AMDGPU_TARGETS := gfx1100
else ifneq ($(filter gfx906,$(AMDGPU_TARGETS)),)
  WAVEFRONTSIZE64 ?= 1
  AMDGPU_TARGETS := gfx906
else ifneq ($(filter gfx90a,$(AMDGPU_TARGETS)),)
  WAVEFRONTSIZE64 ?= 1
  AMDGPU_TARGETS := gfx90a
else ifneq ($(filter gfx942,$(AMDGPU_TARGETS)),)
  WAVEFRONTSIZE64 ?= 1
  AMDGPU_TARGETS := gfx942
else
  $(warning Did not find a supported AMD device. Rebuild with AMDGPU_TARGETS env variable to force build for device)
endif
ifndef MULTI_GPU # use MULTI_GPU to force a multi-gpu build in a cross compile situation
  ifeq ($(shell test `$(ROCM_PATH)/llvm/bin/amdgpu-offload-arch -a | grep $(AMDGPU_TARGETS) | wc -l` -lt 2; echo $$?),0)
    NO_MULTI_GPU ?= 1
  endif
endif
HIPCC_FLAGS += $(addprefix --offload-arch=,$(AMDGPU_TARGETS))
ifneq ($(NO_MULTI_GPU), 1)
  ifdef RCCL_PATH
    HIPCC_FLAGS += -I$(RCCL_PATH)/include
    HIPCC_LDFLAGS += -L$(RCCL_PATH)
  endif
  ifeq ($(shell [ -d /usr/lib/x86_64-linux-gnu/openmpi/lib/ ] && [ -d /usr/lib/x86_64-linux-gnu/openmpi/include/ ] && echo "exists"), exists)
    HIPCC_FLAGS += -I/usr/lib/x86_64-linux-gnu/openmpi/include -DMULTI_GPU -DUSE_MPI
    HIPCC_LDFLAGS += -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ -lmpi -lrccl
  endif
endif
ifdef HIPBLASLT_PATH
  HIPCC_FLAGS += -I$(HIPBLASLT_PATH)/include
  HIPCC_LDFLAGS += -L$(HIPBLASLT_PATH)/lib
endif
ifdef HIPBLAS_PATH
  HIPCC_FLAGS += -I$(HIPBLAS_PATH)/include
  HIPCC_LDFLAGS += -L$(HIPBLAS_PATH)/lib
endif
ifdef WAVEFRONTSIZE64
  HIPCC_FLAGS += -DWAVEFRONTSIZE64 -mwavefrontsize64
endif
AMD_HEADERS = $(addprefix $(BUILD_DIR)/hip/,$(wildcard llmc/*h))

# autodect a lot of various supports on current platform
$(info ---------------------------------------------)

ifneq ($(OS), Windows_NT)
  NVCC := $(shell which nvcc 2>/dev/null)
  NVCC_LDFLAGS += -lnvidia-ml

  # Function to test if the compiler accepts a given flag.
  define check_and_add_flag
    $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
    ifeq ($(FLAG_SUPPORTED),yes)
        CFLAGS += $(1)
    endif
  endef

  # Check each flag and add it if supported
  $(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))
else
  CFLAGS :=
  REMOVE_FILES = del *.exe,*.obj,*.lib,*.exp,*.pdb && del
  SHELL_UNAME := Windows
  ifneq ($(shell where nvcc 2> nul),"")
    NVCC := nvcc
  else
    NVCC :=
  endif
  CC := cl
  CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
   /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
  LDFLAGS :=
  LDLIBS :=
  INCLUDES :=
  NVCC_FLAGS += -I"dev"
  ifeq ($(WIN_CI_BUILD),1)
    $(info Windows CI build)
    OUTPUT_FILE = /link /OUT:$@
    CUDA_OUTPUT_FILE = -o $@
  else
    $(info Windows local build)
    OUTPUT_FILE = /link /OUT:$@ && copy /Y $@ $@.exe
    CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
  endif
endif

# Check and include cudnn if available
# You can override the path to cudnn frontend by setting CUDNN_FRONTEND_PATH on the make command line
# By default, we look for it in HOME/cudnn-frontend/include and ./cudnn-frontend/include
# Refer to the README for cuDNN install instructions
ifeq ($(USE_CUDNN), 1)
  ifeq ($(SHELL_UNAME), Linux)
    ifeq ($(shell [ -d $(HOME)/cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= $(HOME)/cudnn-frontend/include
    else ifeq ($(shell [ -d cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= cudnn-frontend/include
    else
      $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
    endif
    NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
    NVCC_LDFLAGS += -lcudnn
    NVCC_FLAGS += -DENABLE_CUDNN
    NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  else
    ifneq ($(OS), Windows_NT)
      $(info → cuDNN is not supported on MAC OS right now)
    else
      $(info ✓ Windows cuDNN found, will run with flash-attention)
      ifeq ($(shell if exist "$(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= $(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include #override on command line if different location
      else ifeq ($(shell if exist "cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= cudnn-frontend\include #override on command line if different location
      else
        $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
      endif
      CUDNN_INCLUDE_PATH ?= -I"C:\Program Files\NVIDIA\CUDNN\v9.1\include\12.4"
      CUDNN_FRONTEND_PATH += $(CUDNN_INCLUDE_PATH)
      NVCC_FLAGS += --std c++20 -Xcompiler "/std:c++20" -Xcompiler "/EHsc /W0 /nologo /Ox /FS" -maxrregcount=0 --machine 64
      NVCC_CUDNN = $(BUILD_DIR)\cudnn_att.obj
      NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
      NVCC_LDFLAGS += -L"C:\Program Files\NVIDIA\CUDNN\v9.1\lib\12.4\x64" -lcudnn
      NVCC_FLAGS += -DENABLE_CUDNN
    endif
  endif
else
  $(info → cuDNN is manually disabled by default, run make with `USE_CUDNN=1` to try to enable)
endif

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
# First, check if NO_OMP is set to 1, if not, proceed with the OpenMP checks
ifeq ($(NO_OMP), 1)
  $(info OpenMP is manually disabled)
else
  ifneq ($(OS), Windows_NT)
  # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      # Check for Homebrew's libomp installation in different common directories
      ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on ARM (Apple Silicon)
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/opt/homebrew/opt/libomp/include
        $(info ✓ OpenMP found)
      else ifeq ($(shell [ -d /usr/local/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on Intel
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/usr/local/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/usr/local/opt/libomp/include
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    else
      # Check for OpenMP support in GCC or Clang on Linux
      ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
        CFLAGS += -fopenmp -DOMP
        LDLIBS += -lgomp
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    endif
  endif
endif

# Check if NCCL is available, include if so, for multi-GPU training
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU (NCCL) is manually disabled)
else
  ifneq ($(OS), Windows_NT)
    # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      $(info ✗ Multi-GPU on CUDA on Darwin is not supported, skipping NCCL support)
    else ifeq ($(shell dpkg -l | grep -q nccl && echo "exists"), exists)
      $(info ✓ NCCL found, OK to train with multiple GPUs)
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDLIBS += -lnccl
    else
      $(info ✗ NCCL is not found, disabling multi-GPU support)
      $(info ---> On Linux you can try install NCCL with `sudo apt install libnccl2 libnccl-dev`)
    endif
  endif
endif

# Attempt to find and include OpenMPI on the system
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
OPENMPI_LIB_PATH = $(OPENMPI_DIR)/lib/
OPENMPI_INCLUDE_PATH = $(OPENMPI_DIR)/include/
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI is manually disabled)
else ifeq ($(shell [ -d $(OPENMPI_LIB_PATH) ] && [ -d $(OPENMPI_INCLUDE_PATH) ] && echo "exists"), exists)
  $(info ✓ MPI enabled)
  NVCC_INCLUDES += -I$(OPENMPI_INCLUDE_PATH)
  NVCC_LDFLAGS += -L$(OPENMPI_LIB_PATH)
  NVCC_LDLIBS += -lmpi
  NVCC_FLAGS += -DUSE_MPI
else
  $(info ✗ MPI not found)
endif

# Precision settings, default to bf16 but ability to override
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu profile_gpt2cu

# Add targets
TARGETS = train_gpt2 test_gpt2

# Conditional inclusion of CUDA targets
ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU/CUDA builds)
else
    $(info ✓ nvcc found, including GPU/CUDA support)
    TARGETS += train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu $(NVCC_CUDNN)
endif

# Conditional inclusion of AMD targets
ifeq ($(HIPCC),)
    $(info ✗ hipcc not found, skipping GPU/AMD builds)
else
    $(info ✓ hipcc found, building for $(AMDGPU_TARGETS))
    TARGETS += train_gpt2amd test_gpt2amd train_gpt2_fp32amd test_gpt2_fp32amd profile_gpt2amd
    HIPCC_FLAGS += -DBUILD_AMD
endif

$(info ---------------------------------------------)

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2fp32cu: test_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

profile_gpt2cu: profile_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -lineinfo $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS)  $(CUDA_OUTPUT_FILE)

### AMD builds:

$(BUILD_DIR)/hip/llmc/%h: llmc/%h
	@mkdir -p $(dir $@)
	$(HIPIFY) -quiet-warnings $< -o $@

amd_headers: $(AMD_HEADERS)

$(BUILD_DIR)/hip/%.cu: %.cu
	@mkdir -p $(dir $@)
	$(HIPIFY) -quiet-warnings $< -o $@

%amd: $(BUILD_DIR)/hip/%.cu amd_headers
	$(HIPCC) $(HIPCC_FLAGS) $(PFLAGS) $< $(HIPCC_LDFLAGS) -o $@

profile_gpt2amd: $(BUILD_DIR)/hip/profile_gpt2.cu $(BUILD_DIR)/hip/train_gpt2.cu amd_headers
	$(HIPCC) $(HIPCC_FLAGS) $(PFLAGS) $< $(HIPCC_LDFLAGS) -o $@

test_gpt2amd: $(BUILD_DIR)/hip/test_gpt2.cu $(BUILD_DIR)/hip/train_gpt2.cu amd_headers
	$(HIPCC) $(HIPCC_FLAGS) $(PFLAGS) $< $(HIPCC_LDFLAGS) -o $@

clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
	rm -rf $(BUILD_DIR)/hip
