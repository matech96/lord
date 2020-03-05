FROM matech96/jupyter_base

RUN pip3 install keras matplotlib scikit-learn pandas bokeh pydot holoviews librosa scikit-image moviepy tensorflow-graphics-gpu
RUN apt update && apt-get install -y graphviz
RUN pip3 install comet-ml==2.0.15
RUN pip3 install opencv-python morecvutils
RUN apt-get update
RUN apt-get install -y build-essential cmake
#$ sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
#$ sudo apt-get install libx11-dev libgtk-3-dev
RUN pip3 install keras-lr-multiplier dlib
RUN apt update && apt-get install -y git
RUN pip3 install git+https://github.com/alexlee-gk/lpips-tensorflow.git
RUN pip3 install tensorflow-gpu==1.15 tensorflow-datasets