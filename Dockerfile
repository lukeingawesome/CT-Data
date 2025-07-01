# ────────────────────────────────────────────────────────────────
# CUDA 12.2 tool‑chain ‑ Ubuntu 22.04
# ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000
ARG USERNAME=user
ARG PYTHON_VERSION=3.10            # 22.04 ships 3.10 natively
ENV DEBIAN_FRONTEND=noninteractive
ARG PROJECT_ROOT=/opt/project

# ────────────────────────────────────────────────────────────────
# System + Python
# ────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
        python3-venv python3-pip \
        build-essential git curl ca-certificates wget gpg \
        ninja-build cmake \
        libopenmpi-dev openmpi-bin \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        tzdata openssh-client sudo && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────
# Intel oneAPI CCL for DeepSpeed
# ────────────────────────────────────────────────────────────────
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-ccl-devel && \
    rm -rf /var/lib/apt/lists/*

# Set up oneAPI environment
ENV CPATH=/opt/intel/oneapi/ccl/latest/include:$CPATH \
    LIBRARY_PATH=/opt/intel/oneapi/ccl/latest/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/latest/lib:$LD_LIBRARY_PATH

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ────────────────────────────────────────────────────────────────
# Core Python stack (cu121 wheels run fine on CUDA 12.2)
# ────────────────────────────────────────────────────────────────
###############################################################################
# 0 ▸  Common pip flags / cache
###############################################################################
ENV PIP_DEFAULT_TIMEOUT=120          \
    PIP_RETRIES=10                   \
    PIP_NO_INPUT=1                   \
    TORCH_CUDA_ARCH_LIST=8.6         \
    DS_BUILD_OPS=1                   \
    FLASH_ATTENTION_FORCE_CUDA=1

# BuildKit cache keeps partly‑downloaded wheels
# NB: requires Docker 20.10+ with BuildKit (enabled by default on Docker 24)
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

###############################################################################
# 1 ▸  Install PyTorch first (required for deepspeed)
###############################################################################
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 \
        torchvision==0.19.1+cu121 \
        numpy>=1.26.4

###############################################################################
# 2 ▸  Build flash‑attention
###############################################################################
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation \
        flash_attn==2.7.0.post2

###############################################################################
# 3 ▸  DeepSpeed 0.16.1 with CCL support (now that oneCCL is installed)
###############################################################################
ENV DS_BUILD_OPS=1 \
    DS_BUILD_CCL=1 \
    DS_BUILD_AIO=0 \
    DS_BUILD_UTILS=1

RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install --no-build-isolation \
        deepspeed==0.16.1

###############################################################################
# 4 ▸  Everything else (pure‑Python)
###############################################################################
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install \
        "monai>=1.3.0" \
        huggingface_hub \
        "transformers>=4.38.2" \
        nibabel nltk \
        pandas tqdm matplotlib opencv-python \
        timm wandb scikit-image iopath tensorboard


# ────────────────────────────────────────────────────────────────
# CUDA build flags
# ────────────────────────────────────────────────────────────────
ENV TORCH_CUDA_ARCH_LIST=8.6 \
    DS_BUILD_OPS=1 \
    FLASH_ATTENTION_FORCE_CUDA=1

# ────────────────────────────────────────────────────────────────
# Non‑root dev user (matches your host UID/GID)
# ────────────────────────────────────────────────────────────────
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd  -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} && \
    usermod  -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p ${PROJECT_ROOT} \
    && chown -R ${USERNAME}:${USERNAME} ${PROJECT_ROOT} \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# ────────────────────────────────────────────────────────────────
# After creating the non‑root user
# ────────────────────────────────────────────────────────────────
RUN mkdir -p /home/${USERNAME}/.cursor-server \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cursor-server

USER ${USERNAME}
WORKDIR ${PROJECT_ROOT}

# VS Code / Cursor will upload its own server here; no need to pre‑create
CMD [ "/bin/bash", "--login" ]
