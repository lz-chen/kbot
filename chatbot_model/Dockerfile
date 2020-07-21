FROM python:3.7.4
#WORKDIR /media/liah/DATA/docker_image
#COPY . /media/liah/DATA/docker_image
#RUN pip install --trusted-host pypi.python.org -r requirements.txt
#EXPOSE 8000
#ENV PYTHONPATH /media/liah/DATA/docker_image/gpt_model/src:$PYTHONPATH
#CMD ["python", "app.py"]

#FROM tensorflow/tensorflow:1.12.0-gpu-py3

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0" \
    LANG=C.UTF-8

WORKDIR /media/liah/DATA/docker
COPY . /media/liah/DATA/docker
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 8000
ENV PYTHONPATH /media/liah/DATA/docker/gpt_model/src:$PYTHONPATH
CMD ["python", "app.py"]