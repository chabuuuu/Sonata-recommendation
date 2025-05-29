# Giai đoạn 1: Cài đặt Conda và môi trường
FROM continuumio/miniconda3 AS builder
WORKDIR /env
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Giai đoạn 2: Image cuối cùng
FROM openjdk:11-jre-slim
WORKDIR /app
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
COPY . .
CMD ["conda", "run", "--no-capture-output", "-n", "GetTipsRec", "python", "main.py"]