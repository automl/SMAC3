FROM continuumio/miniconda3

RUN apt update && apt install -y make cmake libboost-all-dev build-essential doxygen

WORKDIR /app

RUN /opt/conda/bin/conda install -y python=3.9
ENV PATH=/opt/conda/bin:$PATH
RUN /opt/conda/bin/conda install gxx_linux-64 gcc_linux-64 swig


RUN git clone https://github.com/dengdifan/random_forest_run.git
RUN cd random_forest_run && git checkout rfr_dev  && mkdir build && cd build && cmake .. && make pyrfr_docstrings && cd python_package && pip install . --user

RUN git clone https://github.com/dengdifan/SMAC3
RUN cd SMAC3 && git checkout boing && pip install pip install .[gpytorch]

ENV LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH