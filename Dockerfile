FROM python:3.9-slim

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Build-time environmental variable so that apt doesn't complain
ARG DEBIAN_FRONTEND=noninteractive

RUN cd /usr/local/ && pip3 install --no-cache-dir phylodeep==0.8

# File Author / Maintainer
MAINTAINER Anna Zhukova <anna.zhukova@pasteur.fr>

# Clean up
RUN mkdir /pasteur

# The entrypoint runs the command line
ENTRYPOINT ["/bin/bash"]