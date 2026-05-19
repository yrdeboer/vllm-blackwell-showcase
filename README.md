2026 May 28

# Building and Optimizing vLLM from Source for NVIDIA Blackwell (sm_120) on WSL2

An enterprise-grade deployment showcase demonstrating bare-metal optimization, environment isolation,
and advanced runtime configuration of the vLLM inference engine on the NVIDIA Blackwell architecture.

---

## 1. Introduction & Engineering Objectives

Deploying large-scale language models (LLMs) efficiently requires a deep convergence of hardware awareness, compiler orchestration,
and runtime stability. While low-code API integrations and pre-compiled packages satisfy basic application layers, they abstract
away the performance vectors critical for high-throughput production environments.

This repository serves as a technical showcase for compiling and optimizing the **vLLM** inference engine directly from
source to leverage the native instructions of the **NVIDIA Blackwell architecture (RTX 5090 / sm_120)**. 

### The Engineering Challenges Addressed:
* **Resource-Constrained Compilation:** Navigating the physical memory boundaries of a WSL2 environment (14GB RAM allocated) during intensive CUDA compilation without triggering Out-Of-Memory (OOM) kernel kills.
* **Deterministic Dependency Orchestration:** Mitigating Python package-manager resolution conflicts (PEP 440) that inherently attempt to overwrite specific PyTorch Nightly builds during downstream installation phases.
* **C++ ABI Linkage & Runtime Resolution:** Diagnosing and resolving broken dynamic linkages (`undefined symbol` errors) across the boundary of PyTorch's native shared libraries and vLLM's compiled C++ extensions.
* **VRAM Boundary Management:** Serving a 32-billion parameter model (`Qwen2.5-32B-Instruct-GPTQ-Int4`) within a tight VRAM budget, balancing aggressive KV-cache management with Host OS stability.

By executing this end-to-end framework, the deployment achieves optimized token-per-second throughput while maintaining structural isolation from the underlying host system.

---

## 2. Architectural Stack Overview

The deployment architecture is orchestrated across the following layers:


| Layer | Component | Specification / Version |
| --- | --- | --- |
| **Hardware** | GPU Host | NVIDIA GeForce RTX 5090 (Blackwell Architecture, `sm_120`) |
| **Host OS** | Operating System | Windows 11 Home with WSL2 (Ubuntu 22.04 LTS / 24.04 LTS) |
| **Compiler** | CUDA Toolkit | Version 13.2 (Driver API compliant, Toolkit isolated) |
| **Framework** | PyTorch Backend | Version 2.12.0.dev (Nightly build compiled for `cu128`) |
| **Engine** | Inference Serving | vLLM (Compiled from Source with `sm_120` specific kernels) |
| **Model** | Target LLM | Qwen2.5-32B-Instruct-GPTQ-Int4 |


## 3. Repository Structure

* `README.md` - Technical documentation, architectural rationale, and step-by-step reproduction guide.
* `check_nvcc.cu` - Native C++/CUDA diagnostic script to validate compilation vectors for the `sm_120` virtual architecture.
* `check_torch_on_gpu.py` - PyTorch framework diagnostic verifying Blackwell tensor core communication and runtime integrity.
* `check_deployment.py` - Automated client benchmarking script measuring real-time token throughput and server latency.
* `build_log.txt` - Full output of build
---

# 4. Prepare vLLM Build

## 4.1 Isolation of user space
To keep a clean and reproduceable environment we'll create a new user on the system
and initialise a conda env for it.

We start by creating a new user and logging it in 

    $ sudo adduser ai_architect
	...
    $ sudo usermod -aG sudo ai_architect
    $ su - ai_architect
	$ cd; pwd

For now on showing working directory:

    ai_architect@MSI:~$ pwd
    /home/ai_architect

Install miniconda and init the bash shell

    ai_architect@MSI:~$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ai_architect@MSI:~$ bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    ai_architect@MSI:~$ $HOME/miniconda3/bin/conda init bash
    
    ==> For changes to take effect, close and re-open your current shell. <==
    
    ai_architect@MSI:~$ source ~/.bashrc
    (base) ai_architect@MSI:~$
    (base) ai_architect@MSI:~$ rm Miniconda3-latest-Linux-x86_64.sh
    
## 4.2 Install CUDA toolkit system wide w/o drivers

We need the CUDA compiler nvcc and the header files before installing python packages,
in case they are used during *their* installation.

Note we *do not install* the NVidea drivers on the system, this has been done by Windows and
we will not touch that.

We download specifically the CUDA 13.2 version, which takes a couple of minutes.

    (base) ai_architect@MSI:~$ wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
    2026-05-18 19:36:46 (12.2 MB/s) - `cuda_13.2.1_595.58.03_linux.run' saved [4398952964/4398952964]
	
We set some important variables in the bash environment, that are used throughout.

    (base) ai_architect@MSI:~$ echo 'export CUDA_HOME=/usr/local/cuda-13.2' >> ~/.bashrc
    (base) ai_architect@MSI:~$ echo 'export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}' >> ~/.bashrc
    (base) ai_architect@MSI:~$ echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    (base) ai_architect@MSI:~$ source ~/.bashrc
	(base) ai_architect@MSI:~$ rm cuda_13.2.1_595.58.03_linux.run
	
We test if we can now compile with flag for blackwell (sm_120) architecture (C++ file in repo).

    (base) ai_architect@MSI:~$ nano check_nvcc.cu
    (base) ai_architect@MSI:~$ nvcc -arch=sm_120 check_nvcc.cu -o check_nvcc
    (base) ai_architect@MSI:~$ ./check_nvcc
    System architecture validation OK.
    Number of devices detected by CUDA: 1
    
## 4.3 Python runtime and pytorch configuration

We will create a python runtime with conda.

    (base) ai_architect@MSI:~$ conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    (base) ai_architect@MSI:~$ conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    accepted Terms of Service for https://repo.anaconda.com/pkgs/main
    accepted Terms of Service for https://repo.anaconda.com/pkgs/r
    (base) ai_architect@MSI:~$ conda create -n blackwell-vllm-core python=3.11 -y
    (base) ai_architect@MSI:~$ conda activate blackwell-vllm-core
    (blackwell-vllm-core) ai_architect@MSI:~$
    
We will have to install a specific version of pytorch for the blackwell architecture, note that is was built against CUDA 12.8.
This takes a couple of minutes.

    (blackwell-vllm-core) ai_architect@MSI:~$ pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
	(blackwell-vllm-core) ai_architect@MSI:~$ pip install numpy
	
Test this (python file in repo)

    (blackwell-vllm-core) ai_architect@MSI:~$ python check_torch_on_gpu.py
    Active Torch: 2.12.0.dev20260408+cu128
    CUDA-version used by Torch: 12.8
    Blackwell (sm_120) support: Yes

# 5. Controlled compilation of vLLM for blackwell
We have a working C++ compiler and a python runtime with proper PyTorch.

## 5.1 Get the vLLM source code
    (blackwell-vllm-core) ai_architect@MSI:~$ git clone https://github.com/vllm-project/vllm.git
	(blackwell-vllm-core) ai_architect@MSI:~$ cd vllm/
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ git rev-parse HEAD
    a2c8fc66573664395f491a94da1882fdf92e034b

## 5.2 Set build flags
We need to set some flags to ensure compilation against blackwell.
We put the MAX_JOBS to 1 because we have limited RAM, safe side.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ export TORCH_CUDA_ARCH_LIST="12.0"
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ export MAX_JOBS=1

Then we need to use the vllm utility to change config for building against an existing pytorch.
And also install some more tools for the build.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ python use_existing_torch.py
	(blackwell-vllm-core) ai_architect@MSI:~/vllm$ pip install -r requirements/build/cuda.txt

## 5.3 Handle PyTorch version override by pip
We need to take care that our existing pytorch will not be replaced by a "stable" pytorch by pip, since we are
using a nightly version.


    (blackwell-vllm-core) ai_architect@MSI:~$ echo "torch==2.12.0.dev20260408+cu128" > /tmp/vllm_constraints.txt
    (blackwell-vllm-core) ai_architect@MSI:~$ export PIP_CONSTRAINT=/tmp/vllm_constraints.txt

Now we get to the actual build, which took 3-4 hours.
It will compile the C++/CUDA extensions for the RTX-5090 GPU and update the python runtime where necessary.
The build log is in the repo.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ pip install -e . --no-build-isolation -v &> build_log.txt &
	
	
# 6. ABI linkage diagnosis and runtime evaluation
## 6.1 Checking versions and paths
Checking the torch version:

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ python -c "import torch; print(torch.__path__[0]); print(torch.__version__)"
    /home/ai_architect/miniconda3/envs/blackwell-vllm-core/lib/python3.11/site-packages/torch
    2.12.0.dev20260408+cu128
    
OK.

Checking the vllm version.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ python -c "import vllm; print(vllm.__path__[0]); print(vllm.__version__)"
    /home/ai_architect/vllm/vllm
    0.21.1rc1.dev76+ga2c8fc665.d20260518
    
OK.

Checking vllm binary links:

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ ldd `echo $(python -c "import vllm; print(vllm.__path__[0])")/_C.abi3.so` | grep 'not found'
            libtorch.so => not found
            libtorch_cpu.so => not found
            libtorch_cuda.so => not found
            libc10_cuda.so => not found
            libc10.so => not found
    
Not OK, libs are in conda venv and cannot be found through normal paths.

But wait, the `import vllm` worked fine! Probably pytorch is loaded first, after which the shared libs become findable *at runtime*.

Let's see if a mere setting of the link LD library path would resolve the "not founds":

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ echo $(python -c "import torch; import os; print(os.path.join(torch.__path__[0], \"lib\"))")
    /home/ai_architect/miniconda3/envs/blackwell-vllm-core/lib/python3.11/site-packages/torch/lib
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ echo $LD_LIBRARY_PATH
    /usr/local/cuda-13.2/lib64
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ai_architect/miniconda3/envs/blackwell-vllm-core/lib/python3.11/site-packages/torch/lib
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ ldd `echo $(python -c "import vllm; print(vllm.__path__[0])")/_C.abi3.so` | grep 'not found'
    
OK, temporary fix would work, we could put this in the `~/.bashrc`.

But ... I rather keep conda env specifics outside of the `~/.bashrc`.
So I'll patch the binary to set a pointer there, which is less work than adjusting the
conda env activation/deactivation scripts to update the LD_LIBRARY_PATH.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))")
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ VLLM_SO=$(python -c "import vllm; import os; print(os.path.join(vllm.__path__[0], '_C.abi3.so'))")
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ ldd $VLLM_SO | grep "not found"
            libtorch.so => not found
            libtorch_cpu.so => not found
            libtorch_cuda.so => not found
            libc10_cuda.so => not found
            libc10.so => not found
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ sudo apt-get update
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ sudo apt-get install patchelf -y			
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ patchelf --set-rpath $TORCH_LIB $VLLM_SO
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ ldd $VLLM_SO | grep "not found"
    (blackwell-vllm-core) ai_architect@MSI:~/vllm$ patchelf --print-rpath $VLLM_SO
    /home/ai_architect/miniconda3/envs/blackwell-vllm-core/lib/python3.11/site-packages/torch/lib

OK.

## 6.2 Runtime evaluation -- vllm package shadowing
Now we have the `vllm` repo map in our home dir. This leads to package shadowing, for which
we were warned during the vllm build:

          ********************************************************************************
          Please be careful with folders in your working directory with the same
          name as your package as they may take precedence during imports.
          ********************************************************************************
		  
So we move all our files scripts to a new map, which will be the working map and this repo.

    (blackwell-vllm-core) ai_architect@MSI:~$ mkdir vllm-blackwell-showcase
    (blackwell-vllm-core) ai_architect@MSI:~$ mv build_log.txt check_nvcc.cu check_torch_on_gpu.py vllm-blackwell-showcase/


## 6.3 Runtime evaluation -- get the model
We are going to use `Qwen2.5-32B-Instruct-GPTQ-Int4`. 
This is quantised to 4 bits so it will fit onty my GPU, which has 24GB VRAM, with space left for some context length.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm-blackwell-showcase$ pip install "huggingface_hub[cli]"
    (blackwell-vllm-core) ai_architect@MSI:~/vllm-blackwell-showcase$ hf download Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --local-dir /home/ai_architect/models/qwen-32b-gptq
	
OK

## 6.4 Runtime evaluation -- VRAM usage considerations and runtime flags
We strive to have a deployment that will not destabilise my laptop, so we leave 3 GB of VRAM for the host system. 
That should be enough, since for normal operations I've never seen it go much over the 2 GB --> fix to 88% usage of VRAM.

	Model usage = 32.5 *1e9 parameters * 0.5 (4 bits each) = 15.1 GB <-- yeah, but from previous runs I know this is gonna be 18GB.
	Overhead (CUDA graph): 1 GB
    2 (Key + Value matrices) * 8 (num heads in this model) * 128 (head size) * 64 (layer count) * 1 (for byte size floats) = 131072 bytes per input token.
    Available for KV-cache: (20 - 18 - 1) = 1GB 
    Max context length: (20 - 15.1 - 1) * 1024**3 / 131072 = 8192 tokens
	
Setting the context length to 4096 should give ample space.

| **Flag name** | **Value** | **Considerations** |
| ---       |  ---  | --- |
| --quantization         | gptq_marlin | Best available would probably otherwise use exllama from model config |
| --dtype                | bfloat16    | "auto" would resolve to float16 as from model config |
| --kv-cache-dtype fp8   | fp8         | Halves KV-cache usage, 8 bits is fine |
| --enforce-eager        | Don't use   | It would save VRAM, but omit CUDA graphs |
| --trust-remote-code    | Don't use   | Not needed, model is local and vLLM has Qwen2ForCausalLM |
| --max-model-len        | 4092        | As estimated above |
| --gpu-memory-utilization | 0.88       | To leave enough VRAM for other apps |



## 6.5 Runtime evaluation -- running the server
Don't start it from the home dir, the python importer will search in the vllm repo map and not find what it is looking for.

    (blackwell-vllm-core) ai_architect@MSI:~/vllm-blackwell-showcase$ python -m vllm.entrypoints.openai.api_server \
        --model /home/ai_architect/models/qwen-32b-gptq \
        --quantization gptq_marlin \
        --kv-cache-dtype fp8 \
        --dtype bfloat16 \
        --port 8000 \
        --gpu-memory-utilization 0.88 \
        --max-model-len 4096

The output of this command is in `startup_log.txt`.

We highlight:

    (EngineCore pid=23199) INFO 05-19 11:21:39 [monitor.py:53] torch.compile took 0.56 s in total
	
That is a very quick start.

    (EngineCore pid=23199) INFO 05-19 11:21:44 [kernel_warmup.py:69] Warming up FlashInfer attention.
    (EngineCore pid=23199) INFO 05-19 11:22:01 [gpu_model_runner.py:6416] Graph capturing finished in 15 secs, took 0.95 GiB

FlashInfer is new and much faster than default Triton alternatives so we expect maximum throughput.

    (EngineCore pid=23199) INFO 05-19 11:21:44 [kv_cache_utils.py:1733] GPU KV cache size: 9,584 tokens
    (EngineCore pid=23199) INFO 05-19 11:21:44 [kv_cache_utils.py:1734] Maximum concurrency for 4,096 tokens per request: 2.34x
	
Here we see we have some space left, we could double the context size! Or we leave it like this and we can serve 2 requests at once.

## 6.6 Runtime evaluation -- prompting the model
In another shell:

    (blackwell-vllm-core) ai_architect@MSI:~/vllm-blackwell-showcase$ python check_deployment.py  /home/ai_architect/models/qwen-32b-gptq
    Sending prompt: 'Describe the 5 most important elements in the EU AI Act using max 50 words for each.'...
    Got result:
    1. **Risk-Based Approach**: The Act categorizes AI systems based on risk levels, from minimal to unacceptable, ensuring stringent oversight for high-risk applications like facial recognition and biometric identification, aiming to protect citizens' rights and safety.
    
    2. **Transparency Requirements**: High-risk AI systems must provide clear information about their capabilities, limitations, and risks, ensuring users understand how these technologies operate and can make informed decisions.
    
    3. **Human Oversight**: Mandates that AI systems allow for human intervention and control, preventing autonomous decision-making that could lead to harmful outcomes, ensuring accountability and ethical use of technology.
    
    4. **Data Governance**: Emphasizes the need for high-quality, unbiased data sets to train AI models, aiming to prevent discrimination and ensure fair outcomes across all demographics, promoting equality and justice.
    
    5. **Market Surveillance**: Establishes mechanisms for monitoring and enforcing compliance with AI regulations, including regular audits and penalties for non-compliance, ensuring the safe and lawful operation of AI systems within the EU.
    ================
    Inference duration: 6.86 seconds.
    Average throughput: 22.44 tokens/s.
    
Output from the server shell:

    (EngineCore pid=23199) WARNING 05-19 11:48:22 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
    (APIServer pid=23075) INFO:     127.0.0.1:39786 - "POST /v1/chat/completions HTTP/1.1" 200 OK
    (APIServer pid=23075) INFO 05-19 11:48:32 [loggers.py:271] Engine 000: Avg prompt throughput: 5.0 tokens/s, Avg generation throughput: 20.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
    (APIServer pid=23075) INFO 05-19 11:48:42 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
    
Note the warning. We send off the same query again to see the warmed up throughput and cache usage:

    Inference duration: 5.97 seconds.
    Average throughput: 24.63 tokens/s.
	
    (APIServer pid=23075) INFO:     127.0.0.1:48360 - "POST /v1/chat/completions HTTP/1.1" 200 OK
    (APIServer pid=23075) INFO 05-19 11:52:02 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 9.9 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 48.0%
    (APIServer pid=23075) INFO 05-19 11:52:12 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 48.0%

OK.

### Production Benchmark Metrics (Qwen2.5-32B-Instruct-GPTQ-Int4)

| Metric | Measured Value | Architectural Context |
| :--- | :--- | :--- |
| **Time-To-First-Token (TTFT)** | <1 seconds | Cold hit including Triton JIT `_compute_slot_mapping` compilation. |
| **Avg Prompt (Prefill) Speed** | 5.0 tokens/s | Single-stream batch-1 constrained by runtime kernel generation. |
| **Avg Decode (Generation) Speed** | 20.3 tokens/s | Optimized execution via custom Blackwell Marlin-linear kernels. |
| **Effective User Throughput** | 22.0 tokens/s | Multi-token aggregate under resource-isolated WSL2 constraints. |

## 6.7 Shutdown
We shutdown by sending SIGTERM:

    (blackwell-vllm-core) ai_architect@MSI:~/vllm-blackwell-showcase$ pkill -15 -f "vllm.entrypoints.openai.api_server"
	
And we get this warning:

    [rank0]:[W519 11:59:41.079346057 ProcessGroupNCCL.cpp:1648] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
	
Next time we will start with `--shutdown-timeout 10` to see if that runs the garbage collector. For now we do not notice any funny system behaviour.

# 7. Disclaimer
Gemini was my copilot, but not my autopilot.

