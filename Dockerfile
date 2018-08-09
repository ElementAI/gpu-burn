FROM nvidia/cuda:9.0-devel

MAINTAINER bgs
WORKDIR /root

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ElementAI/gpu-burn.git && cd gpu-burn && make
WORKDIR /root/gpu-burn

ENTRYPOINT [ "/root/gpu-burn/gpu_burn" ]
# burn for 10 secs
CMD [ "10" ]

