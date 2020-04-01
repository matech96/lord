FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN pip3 install keras matplotlib scikit-learn pandas bokeh pydot holoviews librosa scikit-image moviepy tensorflow-graphics-gpu
RUN apt update && apt-get install -y graphviz
RUN pip3 install opencv-python morecvutils
RUN pip3 install keras-lr-multiplier
RUN pip3 install tensorflow-datasets
RUN pip3 install wandb