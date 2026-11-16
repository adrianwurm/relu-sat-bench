# alpha,beta-CROWN + relu-sat-bench runner.
#
# Build (from the repository root; CPU version, about 2-3 GB):
#   docker build -f docker/abcrown/Dockerfile -t relu-sat-bench/abcrown .
# GPU version (NVIDIA GPU, larger image):
#   docker build -f docker/abcrown/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t relu-sat-bench/abcrown-gpu .
#
# Run (PowerShell; results go to <folder>/results/abcrown-bab.csv):
#   docker run --rm --cpus 4 -v "${PWD}\instances\pilot:/bench" relu-sat-bench/abcrown --bench /bench --tool abcrown-bab
# On the GPU image add:  --gpus all -e ABCROWN_DEVICE=cuda
# The MIP mode (--tool abcrown-mip) needs Gurobi. The gurobipy package brings a size-limited
# license (small models only); for larger instances mount a license file:
#   -v "C:\path\to\gurobi.lic:/opt/gurobi/gurobi.lic" -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# The alpha,beta-CROWN version we tested against (repository state of 10 Oct 2026).
ARG ABCROWN_REF=b58d83c
RUN git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git /opt/abcrown \
 && cd /opt/abcrown \
 && git checkout ${ABCROWN_REF} \
 && git submodule update --init --recursive

# PyTorch first, from the index that matches the hardware (CPU by default), then the rest
# of alpha,beta-CROWN's dependencies from its pyproject.toml (torch==2.11.0 is then satisfied).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.11.0 torchvision --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir /opt/abcrown

ENV ABCROWN_HOME=/opt/abcrown \
    ABCROWN_DEVICE=cpu

WORKDIR /opt/relu-sat-bench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "relu_sat_bench.run"]
