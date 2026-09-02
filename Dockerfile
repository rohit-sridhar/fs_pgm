FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
WORKDIR /data/hmm_modeling/fs_pgm
RUN mkdir /root/.tmp

# add compiler tools and multi arch support
RUN dpkg --add-architecture i386
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    build-essential gcc-multilib g++-multilib \
    git jq vim autoconf gdb curl \
    libx11-dev:i386 libx11-dev libc6-dev-i386 libc6-dev

# Install core prerequisites
RUN apt-get install -y --no-install-recommends \
    software-properties-common cmake \
    gpg-agent

# Install python libraries
RUN pip install --no-cache-dir -e -t /data/hmm_modeling/fs_pgm

# Install copilot
RUN curl -fsSL https://gh.io/copilot-install | bash

CMD ["bash"]

