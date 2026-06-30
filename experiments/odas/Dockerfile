# arm64 Linux build/run env for chatak-odas.
# Matches the aarch64 Raspberry Pi target, so the bundled aarch64 TFLite C lib
# (third_party/tflite/aarch64/libtensorflowlite_c.so) is used as-is — no TFLite build.
# Repo is bind-mounted at /work (not COPYed) so build artifacts land on the host.
FROM arm64v8/debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential pkg-config git \
        libfftw3-dev libconfig-dev libasound2-dev libpulse-dev libjson-c-dev \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
