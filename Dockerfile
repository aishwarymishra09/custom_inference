FROM runpod/pytorch:3.10-2.0.0-117

WORKDIR /workspace

# SHELL ["/bin/bash", "-c"]

# Update and upgrade the system packages (Worker Template)
# Install missing dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils zstd python3.10-venv git-lfs unzip && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

# Fetch the model
COPY builder/test_8_gb_process.py /test_8_gb_process.py
CMD python /test_8_gb_process.py
RUN rm /test_8_gb_process.py

COPY builder/download.py /download.py
CMD python /download.py
RUN rm /download.py
# Add src files (Worker Template)
ADD src .

ENV RUNPOD_DEBUG_LEVEL=INFO
ENV DEBIAN_FRONTEND noninteractive

ENV PYTHONUNBUFFERED=1
CMD python -u /rp_handler.py
