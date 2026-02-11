# Brian-QARI Robot Container
# Runs on the robot's onboard computer
# Minimal footprint: HAL, safety zones 1-3, communication, fallback
# NO AI inference - all intelligence runs in the brain container

FROM python:3.13-slim

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Minimal Python dependencies (NO torch, NO transformers, NO qiskit)
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    grpcio>=1.62.0 \
    protobuf>=4.25.0 \
    opencv-python-headless>=4.9.0 \
    cryptography>=42.0.0 \
    pyyaml>=6.0

# Copy only robot-side code
COPY brian/hal/ brian/hal/
COPY brian/safety/zones/ brian/safety/zones/
COPY brian/safety/isolation/ brian/safety/isolation/
COPY brian/safety/healing/ brian/safety/healing/
COPY brian/communication/ brian/communication/
COPY proto/ proto/
COPY config/ config/

# AI Isolation: Verify no AI libraries are present
RUN python -c "import sys; \
    blocked=['torch','tensorflow','transformers','qiskit','cirq','pennylane']; \
    found=[m for m in blocked if m in sys.modules]; \
    assert not found, f'AI libraries detected on robot: {found}'"

EXPOSE 50051 50052 50053

ENV PYTHONPATH=/app
ENV BRIAN_ROBOT_MODE=true

CMD ["python", "-m", "brian.hal.platforms.generic_ros2"]
