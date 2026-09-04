# Probe-only node image. Build from inside this folder, which is the whole build context:
#
#   docker build -t arge23/gpu-probe:latest probe_node/
#   docker run --gpus all -e HF_TOKEN=... -e HF_REPO_ID=... arge23/gpu-probe:latest
#
# Unlike the parent repo's Dockerfile this does NOT git clone -- the folder is self-contained, so
# the code that runs is the code you built, with no push-then-CACHE_BUST dance and no way to
# silently ship an older commit.
FROM python:3.12-slim

# Stream stdout/stderr instead of buffering, so a node's logs appear live.
ENV PYTHONUNBUFFERED=1

# Dependencies before the source, keyed only on requirements.txt. That layer is several GB of
# torch and its CUDA libraries and is unshared between images; keyed this way it survives every
# code change, so rebuilding after an edit costs kilobytes instead of a fresh multi-GB layer.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "run.py"]
