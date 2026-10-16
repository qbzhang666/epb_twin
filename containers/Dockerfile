# epb_twin with the finite element stack, pinned to the tested versions.
#   docker build -t epb-twin -f containers/Dockerfile .
#   docker run --rm -v "$PWD/my_project:/work" -w /work epb-twin python -m epb_twin check --project project.yaml
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/epb_twin
COPY . .
RUN pip install --no-cache-dir ".[fe,excel]" -c constraints-fe.txt
ENV OMP_WAIT_POLICY=passive
WORKDIR /work
CMD ["python", "-m", "epb_twin", "--help"]
