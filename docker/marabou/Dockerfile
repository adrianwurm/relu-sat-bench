# Marabou (maraboupy 2.0.0) + relu-sat-bench runner.
#
# Build (from the repository root):
#   docker build -f docker/marabou/Dockerfile -t relu-sat-bench/marabou .
# Run on a benchmark folder (PowerShell; results go to <folder>/results/marabou.csv):
#   docker run --rm --cpus 4 -v "${PWD}\instances\pilot:/bench" relu-sat-bench/marabou --bench /bench --tool marabou

# maraboupy has wheels for Linux and Python 3.8-3.11 only.
FROM python:3.11-slim

RUN pip install --no-cache-dir maraboupy==2.0.0

WORKDIR /opt/relu-sat-bench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "relu_sat_bench.run"]
