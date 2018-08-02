FROM nvidia/cuda:9.0-devel

MAINTAINER bgs
WORKDIR /root

#RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
#RUN wget http://wili.cc/blog/entries/gpu-burn/gpu_burn-0.7.tar.gz && tar xzf gpu_burn-0.7.tar.gz && make

RUN apt-get update && apt-install -y git
RUN git clone https://github.com/ElementAI/gpu-burn.git

#ENTRYPOINT [ "/root/gpu_burn" ]
#CMD [ "10" ]   # burn for 10 secs


