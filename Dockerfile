# Sử dụng một image cơ sở có Conda
FROM continuumio/miniconda3

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file environment.yml vào container
COPY environment.yml .

# Tạo môi trường Conda và kích hoạt
RUN conda env create -f environment.yml && conda clean -afy

# Kích hoạt môi trường theo mặc định
RUN echo "conda activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Khai báo cổng (nếu ứng dụng cần)
EXPOSE 5000

# Command để chạy ứng dụng
CMD ["python", "main.py"]
