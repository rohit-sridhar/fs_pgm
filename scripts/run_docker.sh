#!/usr/bin/env bash
set -euo pipefail

# Usage: run_docker.sh <cls|ils|run|launch|rm|build|enter> [CONTAINER_NAME] [SCRIPT_FOR_run]
# Environment variables (optional): LOCAL_HTK_IMAGE, PROJECTS_ROOT, FINGERSPELLING_PATH, FS_TRANSFORMERS_PATH, FINGERSPELLING_VIDEO_PATH, ISLR_MPUTILS_OUT_PATH, VIMRC_FILE

usage() {
    echo "Usage: $0 <cls|ils|run|launch|rm|build|enter> [CONTAINER_NAME] [SCRIPT_FOR_run] [SCRIPT_ARGS...]"
    echo ""
    echo "Notes:"
    echo "  - For 'run', provide SCRIPT_FOR_run; any additional args after the script are forwarded to the script inside the container."
    echo "  - PROJECTS_ROOT is hardcoded in this script (default: /data)."
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

cmd="$1"
container="${2:-}"
if [ "$cmd" = "run" ]; then
    script="${3:-}"
    # capture additional args after the script and forward them to the command run inside container
    extra_args="${@:4}"
fi

local_image="rohit_hmm_fingerspelling"

PROJECTS_ROOT="${PROJECTS_ROOT:-/data}"
LOCAL_HTK_IMAGE="${LOCAL_HTK_IMAGE:-fs_pgm}"

fingerspelling_path="${FINGERSPELLING_PATH:-${PROJECTS_ROOT}/hmm_modeling/fs_pgm}"
fs_transformers_path="${FS_TRANSFORMERS_PATH:-${PROJECTS_ROOT}/deep_learning/fs_transformers}"
fingerspelling_video_path="${FINGERSPELLING_VIDEO_PATH:-${PROJECTS_ROOT}/sign_language_videos/fingerspelling_videos}"
islr_mputils_out_path="${ISLR_MPUTILS_OUT_PATH:-${PROJECTS_ROOT}/deep_learning/ISLR-ML/mputils/out}"
vimrc_dir="${VIM_DIR:-${HOME}/.vim}"

case "$cmd" in
    ils)
        docker image ls
        ;;
    cls)
        docker ps -a --format '{{.Names}}\t{{.Status}}'
        ;;
    build)
        docker build --no-cache -t $local_image .
        ;;
    enter)
        if [ -z "$container" ]; then
            echo "Specify container name for run"
            usage
        fi
        
        docker exec -it "$container" /bin/bash
        ;;
    launch)
        if [ -z "$container" ]; then
            echo "Specify container name when calling launch"
            usage
        fi

        # --platform linux/arm64 \
        docker run -it --rm \
          -v "${fingerspelling_path}":"/data/hmm_modeling/fs_pgm" \
          -v "${fs_transformers_path}":"/data/deep_learning/fs_transformers" \
          -v "${fingerspelling_video_path}":"/data/sign_language_videos/fingerspelling_videos:ro" \
          -v "${islr_mputils_out_path}":"/data/deep_learning/ISLR-ML/mputils:ro" \
          -v "${vimrc_dir}":"/root/.vim" \
          -e HOSTNAME_SERVER="$HOSTNAME" \
          --name "$container" "$LOCAL_HTK_IMAGE"
        ;;

    run)
        if [ -z "$container" ]; then
            echo "Specify container name when calling run"
            usage
        fi
        if [ -z "${script:-}" ]; then
            echo "Specify script path (relative to container WORKDIR) as the third argument"
            usage
        fi

        docker run --rm \
          -v "${fingerspelling_path}":"/data/hmm_modeling/fs_pgm" \
          -v "${fs_transformers_path}":"/data/deep_learning/fs_transformers" \
          -v "${fingerspelling_video_path}":"/data/sign_language_videos/fingerspelling_videos:ro" \
          -v "${islr_mputils_out_path}":"/data/deep_learning/ISLR-ML/mputils:ro" \
          -v "${vimrc_dir}":"/root/.vim" \
          -e HOSTNAME_SERVER="$HOSTNAME" \
          --name "$container" "$LOCAL_HTK_IMAGE" bash -lc "${script} ${extra_args[@]}"
        ;;
    rm)
        if [ -z "$container" ]; then
            echo "Specify container name for rm"
            usage
        fi
        
        docker stop "${container}"
        # docker rm "${container}"
        ;;
    *)
        usage
        ;;
esac

