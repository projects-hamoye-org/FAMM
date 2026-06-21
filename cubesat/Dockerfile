"""Phase 3 Benchmarking Container 
Simulates CubeSat hardware constraints via Docker resource limits."""


FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_cubesat.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy what Phase 3 needs
COPY models/ models/
COPY scripts/benchmark_constrained.py scripts/benchmark_constrained.py
COPY data/results/ data/results/
# copy TIF files for accuracy comparison
COPY data/tif_input/ data/tif_input/

CMD ["python", "scripts/benchmark_constrained.py"]