FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt update && apt-get install -y git
RUN git clone https://github.com/google-research/disentanglement_lib.git
RUN pip install disentanglement_lib[tf_gpu]
RUN dlib_download_data
RUN export DISENTANGLEMENT_LIB_DATA=$(pwd)
