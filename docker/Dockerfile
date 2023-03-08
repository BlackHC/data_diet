FROM nvcr.io/nvidia/tensorflow:23.01-tf2-py3

RUN apt-get update && \
    DEBIAN_FRONTEND=DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        locales \
        rsync \
        software-properties-common \
        tree \
        unzip \
        vim \
        wget \
        && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    pip install --upgrade pip && \
    pip install --upgrade \
        "setuptools<66" \
        && \
    pip install --upgrade \
        numpy\
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        tqdm \
        Cython \
        jupyter \
        jupyterlab \
        tensorflow-datasets \
    && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade \
        "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        neural-tangents \
        flax \
        optax \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /data_diet

EXPOSE 2222
