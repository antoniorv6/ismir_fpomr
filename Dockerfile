FROM pytorch/pytorch:latest

RUN apt update 
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y

RUN pip install --upgrade pip

RUN pip install numpy
RUN pip install opencv-python
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install tensorflow_addons
RUN pip install editdistance
RUN pip install torchinfo