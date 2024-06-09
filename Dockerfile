# Use the official PyTorch image with CUDA 12.1 as the base
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Copy the environment.yaml file
COPY environment.yaml /tmp/environment.yaml

# Install dependencies using micromamba
RUN apt-get update && apt-get install -y curl bzip2 && \
    curl -L -O "https://micro.mamba.pm/api/micromamba/linux-64/latest" && \
    tar -xvf latest -C /usr/local/bin --strip-components=1 && \
    rm latest && \
    micromamba install -y -n base -c conda-forge --file /tmp/environment.yaml && \
    micromamba clean --all --yes && \
    apt-get install -y supervisor

# Set the working directory
WORKDIR /app

# Copy the source code to the working directory
COPY . /app

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Start supervisord
CMD ["/usr/bin/supervisord"]