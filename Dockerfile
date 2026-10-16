# Use the identical base image specified in your Apptainer recipe
FROM python:3.12-slim-trixie

# Replaces %arguments and enforces a non-interactive frontend during build-time
ARG DEBIAN_FRONTEND=noninteractive

# Replaces %environment system variables
ENV TZ=Etc/UTC
ENV PATH="/root/.local/bin:${PATH}"

# Set the working directory (corresponds to your preferred PWD landing directory)
WORKDIR /data/hmm_modeling/fs_pgm

# Replaces the %post section instructions
RUN mkdir -p /root/.tmp && \
    apt-get update -y && apt-get upgrade -y && \
    \
    # Install development toolchain & architecture libraries
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc-14 \
        g++-14 \
        git \
        jq \
        vim \
        autoconf \
        gdb \
        curl \
        tree \
        libx11-dev \
        libc6-dev \
        cmake \
        gpg-agent \
        libeigen3-dev && \
    \
    # Install GitHub Copilot CLI
    curl -fsSL https://gh.io/copilot-install | bash && \
    \
    # Clean up apt caches to minimize container image footprint
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Replaces %runscript to launch a default interactive bash shell
CMD ["bash"]

