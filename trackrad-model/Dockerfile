FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

RUN echo ${TARGETOS}-${TARGETARCH}${TARGETVARIANT}

# TODO: Remove
ENV PYTHONHTTPSVERIFY=0
RUN echo "Acquire::https::Verify-Peer false;" > /etc/apt/apt.conf.d/99disable-https-verify

# noninteractive helps with some prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    build-essential

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Copy pyproject and uv.lock for transitive dependencies
COPY ./pyproject.toml ./uv.lock /app/

WORKDIR /app
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN chown user /app
RUN mkdir /input && chown user /input
RUN mkdir /output && chown user /output
# video group is needed to access GPUs
RUN usermod -a -G video user
USER user

COPY --chown=user:user sam2 /app/sam2/
COPY --chown=user:user nnUNet /app/nnUNet/

RUN uv sync --extra cu124 \
    # TODO: Remove
    --allow-insecure-host https://github.com/ \
    --locked \
    --no-dev \
    --no-cache \
    --compile-bytecode \
    --no-editable

COPY --chown=user:user model.py /app/
COPY --chown=user:user inference.py /app/
COPY --chown=user:user resources /app/resources/

# Ensure that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python","inference.py"]
