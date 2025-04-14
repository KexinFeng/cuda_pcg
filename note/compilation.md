## in setup.py
```
# Use NVCC threads to parallelize the build.
nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
num_threads = min(len(os.sched_getaffinity(0)), nvcc_threads)  
# os.sched_getaffinity(0) is num_cores + 2; ninja -j N  N_default=os.sched_getaffinity(0) + 2
```

## Four cores
```
usage: ninja [options] [targets...]

if targets are unspecified, builds the 'default' target (see manual).

options:
  --version      print ninja version ("1.11.1")
  -v, --verbose  show all command lines while building
  --quiet        don't show progress status, just command output

  -C DIR   change to DIR before doing anything else
  -f FILE  specify input build file [default=build.ninja]

  -j N     run N jobs in parallel (0 means infinity) [default=8 on this system]
  -k N     keep going until N jobs fail (0 means infinity) [default=1]
  -l N     do not start new jobs if the load average is greater than N
  -n       dry run (don't run commands but act like they succeeded)

  -d MODE  enable debugging (use '-d list' to list modes)
  -t TOOL  run a subtool (use '-t list' to list subtools)
    terminates toplevel options; further flags are passed to the tool
  -w FLAG  adjust warnings (use '-w list' to list warnings)

fengx463@agd03 [~/mount_folder/cuda_pcg] % ./cr.sh 
running build_ext
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:448: UserWarning: The detected CUDA version (12.0) has a minor version mismatch with the version that was used to compile PyTorch (12.4). Most likely this shouldn't be a problem.
  warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:458: UserWarning: There are no g++ version bounds defined for CUDA version 12.0
  warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
building 'qed_fermion_module._C' extension
creating /users/4/fengx463/mount_folder/cuda_pcg/build
creating /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9
creating /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:2059: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
Emitting ninja build file /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Using envvar MAX_JOBS (4) as the number of workers...
[1/3] c++ -MMD -MF /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o.d -pthread -B /common/software/install/migrated/anaconda/python3-2021.11-mamba/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /common/software/install/migrated/anaconda/python3-2021.11-mamba/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include -fPIC -O2 -isystem /common/software/install/migrated/anaconda/python3-2021.11-mamba/include -fPIC -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/pybind.cpp -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o -O0 -g -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0
[2/3] /common/software/install/manual/cuda/12.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o.d -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O0 -g -Xcompiler -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89
/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(31): warning #177-D: variable "Ltau" was declared but never referenced

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(34): warning #177-D: variable "stride_vs" was declared but never referenced

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(241): warning #177-D: variable "Ltau" was declared but never referenced

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(244): warning #177-D: variable "stride_vs" was declared but never referenced

[3/3] /common/software/install/manual/cuda/12.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o.d -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O0 -g -Xcompiler -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu: In function ‘at::Tensor mhm_vec(const at::Tensor&, const at::Tensor&, int64_t, float)’:
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu:344:14: warning: narrowing conversion of ‘Ltau’ from ‘int64_t’ {aka ‘long int’} to ‘unsigned int’ [-Wnarrowing]
  344 |     dim3 grid = {Ltau, bs};
      |              ^~~~
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu:344:20: warning: narrowing conversion of ‘bs’ from ‘int64_t’ {aka ‘long int’} to ‘unsigned int’ [-Wnarrowing]
  344 |     dim3 grid = {Ltau, bs};
      |                    ^~
creating build/lib.linux-x86_64-3.9
creating build/lib.linux-x86_64-3.9/qed_fermion_module
g++ -pthread -B /common/software/install/migrated/anaconda/python3-2021.11-mamba/compiler_compat -shared -Wl,-rpath,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath-link,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath-link,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o -L/common/software/install/manual/cuda/12.0/lib64 -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib -L/common/software/install/manual/cuda/12.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-3.9/qed_fermion_module/_C.cpython-39-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.9/qed_fermion_module/_C.cpython-39-x86_64-linux-gnu.so -> qed_fermion_module
Compilation Time: 55.76 seconds
```





## Eight cores

fengx463@agd03 [~/mount_folder/cuda_pcg] % python
Python 3.9.7 (default, Sep 16 2021, 13:09:58) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os
>>> os.sched_getaffinity(0))
  File "<stdin>", line 1
    os.sched_getaffinity(0))
                           ^
SyntaxError: unmatched ')'
>>> os.sched_getaffinity(0)
{0, 1, 2, 3, 4, 65, 66, 67, 68, 69}
>>> ^C

```
fengx463@agd03 [~/mount_folder/cuda_pcg] % ./cr.sh 
usage: ninja [options] [targets...]

if targets are unspecified, builds the 'default' target (see manual).

options:
  --version      print ninja version ("1.11.1")
  -v, --verbose  show all command lines while building
  --quiet        don't show progress status, just command output

  -C DIR   change to DIR before doing anything else
  -f FILE  specify input build file [default=build.ninja]

  -j N     run N jobs in parallel (0 means infinity) [default=12 on this system]
  -k N     keep going until N jobs fail (0 means infinity) [default=1]
  -l N     do not start new jobs if the load average is greater than N
  -n       dry run (don't run commands but act like they succeeded)

  -d MODE  enable debugging (use '-d list' to list modes)
  -t TOOL  run a subtool (use '-t list' to list subtools)
    terminates toplevel options; further flags are passed to the tool
  -w FLAG  adjust warnings (use '-w list' to list warnings)
running build_ext
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:448: UserWarning: The detected CUDA version (12.0) has a minor version mismatch with the version that was used to compile PyTorch (12.4). Most likely this shouldn't be a problem.
  warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:458: UserWarning: There are no g++ version bounds defined for CUDA version 12.0
  warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
building 'qed_fermion_module._C' extension
creating /users/4/fengx463/mount_folder/cuda_pcg/build
creating /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9
creating /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs
/users/4/fengx463/.local/lib/python3.9/site-packages/torch/utils/cpp_extension.py:2059: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
Emitting ninja build file /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Using envvar MAX_JOBS (8) as the number of workers...
[1/3] c++ -MMD -MF /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o.d -pthread -B /common/software/install/migrated/anaconda/python3-2021.11-mamba/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /common/software/install/migrated/anaconda/python3-2021.11-mamba/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include -fPIC -O2 -isystem /common/software/install/migrated/anaconda/python3-2021.11-mamba/include -fPIC -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/pybind.cpp -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o -O0 -g -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0
[2/3] /common/software/install/manual/cuda/12.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o.d -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O0 -g -Xcompiler -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 --threads 8 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89
/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(31): warning #177-D: variable "Ltau" was declared but never referenced

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(34): warning #177-D: variable "stride_vs" was declared but never referenced

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(241): warning #177-D: variable "Ltau" was declared but never referenced

/users/4/fengx463/mount_folder/cuda_pcg/csrs/b_vec_per_tau.cu(244): warning #177-D: variable "stride_vs" was declared but never referenced

[3/3] /common/software/install/manual/cuda/12.0/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o.d -I/common/software/install/manual/cuda/12.0/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/TH -I/users/4/fengx463/.local/lib/python3.9/site-packages/torch/include/THC -I/common/software/install/manual/cuda/12.0/include -I/common/software/install/migrated/anaconda/python3-2021.11-mamba/include/python3.9 -c -c /users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu -o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O0 -g -Xcompiler -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 --threads 8 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu(344): warning #2361-D: invalid narrowing conversion from "signed long" to "unsigned int"

/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu: In function ‘at::Tensor mhm_vec(const at::Tensor&, const at::Tensor&, int64_t, float)’:
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu:344:14: warning: narrowing conversion of ‘Ltau’ from ‘int64_t’ {aka ‘long int’} to ‘unsigned int’ [-Wnarrowing]
  344 |     dim3 grid = {Ltau, bs};
      |              ^~~~
/users/4/fengx463/mount_folder/cuda_pcg/csrs/mhm_vec.cu:344:20: warning: narrowing conversion of ‘bs’ from ‘int64_t’ {aka ‘long int’} to ‘unsigned int’ [-Wnarrowing]
  344 |     dim3 grid = {Ltau, bs};
      |                    ^~
creating build/lib.linux-x86_64-3.9
creating build/lib.linux-x86_64-3.9/qed_fermion_module
g++ -pthread -B /common/software/install/migrated/anaconda/python3-2021.11-mamba/compiler_compat -shared -Wl,-rpath,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath-link,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -Wl,-rpath-link,/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib -L/common/software/install/migrated/anaconda/python3-2021.11-mamba/lib /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/b_vec_per_tau.o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/mhm_vec.o /users/4/fengx463/mount_folder/cuda_pcg/build/temp.linux-x86_64-3.9/csrs/pybind.o -L/common/software/install/manual/cuda/12.0/lib64 -L/users/4/fengx463/.local/lib/python3.9/site-packages/torch/lib -L/common/software/install/manual/cuda/12.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o build/lib.linux-x86_64-3.9/qed_fermion_module/_C.cpython-39-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.9/qed_fermion_module/_C.cpython-39-x86_64-linux-gnu.so -> qed_fermion_module
Compilation Time: 56.85 seconds
```



Debug flags (-O0 -g) disable optimizations but require more metadata generation, which offsets gains from parallelization.

1. Small Project Size Limits Parallelization
Your project has 3 source files (2 CUDA, 1 C++). Even with MAX_JOBS=8:

Ninja can only parallelize compilation of these 3 files (since there are no more independent tasks).

The total compilation time is dominated by the slowest individual file, not the number of cores.

