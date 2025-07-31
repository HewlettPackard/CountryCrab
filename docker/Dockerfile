# create base image that has cuda, python and a working apt
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as ubuntu-cuda-python

ARG PROXY
ENV http_proxy ${PROXY}
ENV https_proxy ${PROXY}
ENV HTTP_PROXY ${PROXY}
ENV HTTPS_PROXY ${PROXY}

RUN if [ "$http_proxy" != "" ]; then \
      echo "Acquire::http::Proxy \"$http_proxy\";" >> /etc/apt/apt.conf; \
    fi

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip

# create the actual development container
FROM ubuntu-cuda-python AS final

# set up sudo
RUN apt-get install -y sudo \
 && groupadd -r wheel \
 && echo "%wheel ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

# additional packages
RUN apt-get install -y \
      curl \
      wget \
      git \
      gh \
      build-essential \
      gcc \
      g++ \
      locales

WORKDIR /tmp

# install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN rm -rf /tmp/*
WORKDIR /

# copy server config and install TLS certificates
WORKDIR /etc/jupyter
COPY docker/jupyter/jupyter_lab_config.py .

WORKDIR /etc/certs
RUN openssl genpkey -algorithm RSA -out jupyter.key \
 && openssl req -new -key jupyter.key -out jupyter.csr -subj "/CN=localhost" \
 && openssl x509 -req -days 365 -in jupyter.csr -signkey jupyter.key -out jupyter.crt

# store user settings here, `dockerctl` can later move them to the correct user's home
COPY docker/jupyter/settings /var/jupyter-settings

WORKDIR /

# ───────────────────────────────────────────────────────────────
# NEW: copy in our entrypoint and make it the default
# ───────────────────────────────────────────────────────────────
# (You’ll create this script at docker/docker-entrypoint.sh)
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# Default to "lab" mode; override by passing "shell" (or any other cmd)
CMD ["lab"]
