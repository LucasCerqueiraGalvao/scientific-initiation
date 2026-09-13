FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

COPY fases/01_validacao_conceitual/requirements.txt /tmp/requirements.txt
COPY fases/01_validacao_conceitual/requirements-comparacao.txt /tmp/requirements-comparacao.txt
COPY fases/01_validacao_conceitual/requirements-modelos.txt /tmp/requirements-modelos.txt
COPY fases/01_validacao_conceitual/requirements-hardware.txt /tmp/requirements-hardware.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements-hardware.txt

ENV PYTHONPATH=/workspace/fases/01_validacao_conceitual

CMD ["python", "-m", "pytest", "fases/01_validacao_conceitual/tests", "-q", "-W", "error"]
