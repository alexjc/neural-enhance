FROM ubuntu:14.04

# Install dependencies
RUN apt-get -qq update           &&  \
    apt-get -qq install --assume-yes \
        "build-essential"            \
        "git"                        \
        "wget"                       \
        "libopenblas-dev"            \
        "liblapack-dev"              \
        "pkg-config"              && \
    rm -rf /var/lib/apt/lists/*

# Miniconda.
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Install requirements before copying project files
WORKDIR /ne
COPY requirements.txt .
RUN /opt/conda/bin/conda install -q -y conda numpy scipy pip pillow
RUN /opt/conda/bin/python3.5 -m pip install -q -r "requirements.txt"

# Copy only required project files
COPY enhance.py .

# Get a pre-trained neural networks, non-commercial & attribution.
RUN wget -q "https://github.com/alexjc/neural-enhance/releases/download/v0.3/ne1x-photo-deblur-0.3.pkl.bz2"
RUN wget -q "https://github.com/alexjc/neural-enhance/releases/download/v0.3/ne1x-photo-repair-0.3.pkl.bz2"
RUN wget -q "https://github.com/alexjc/neural-enhance/releases/download/v0.3/ne2x-photo-default-0.3.pkl.bz2"
RUN wget -q "https://github.com/alexjc/neural-enhance/releases/download/v0.3/ne4x-photo-default-0.3.pkl.bz2"
# Set an entrypoint to the main enhance.py script
ENTRYPOINT ["/opt/conda/bin/python3.5", "enhance.py", "--device=cpu"]
