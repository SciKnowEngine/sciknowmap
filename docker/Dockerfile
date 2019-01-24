FROM ubuntu:16.04
MAINTAINER Gully Burns <gully.burns@chanzuckerberg.com>

# Set the locale
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

# install some basics and libraries needed for dryscrape
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && apt-get update --fix-missing && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 gcc g++ \
    git mercurial subversion xvfb curl vim screen htop less \
    maven sudo ant openjdk-8-jdk -y && \
    # clean up apt \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# set up some basics and add conda to the path for all users
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    mkdir .jupyter && \
    mkdir -p -m 700 .local/share/jupyter && \
    echo "cacert=/etc/ssl/certs/ca-certificates.crt" > .curlrc

# install anaconda
# Remember to download and run Linux versions here.
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

# add conda to the path var
ENV PATH /opt/conda/bin:$PATH
    
RUN conda update -y conda conda-build conda-env
# you can add further conda libraries here \
RUN conda install -y tqdm termcolor joblib \
    # and further conda channels here \
    -c pkgw -c conda-forge -c bioconda
RUN conda clean -i -l -t -y

# add libraries over pip here
RUN pip install textstat git+https://github.com/dask/dask-learn.git wordcloud bottle elasticsearch \
    bottle-cork Click beaker beautifulsoup4 pandas && \
    rm -rf /root/.cache/pip/*

RUN wget --quiet http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz -O /tmp/mallet.tar.gz && \
    cd /tmp && tar -zxvf mallet.tar.gz && rm mallet.tar.gz && cd mallet-2.0.8 && ant
ENV MALLET_HOME /tmp/mallet-2.0.8/

#
# Try to emulate the user running the docker image within the image itself.
#
ARG USER_ID
ARG USER_NAME
RUN echo $USER_ID
RUN useradd -s /bin/bash -m $USER_NAME
RUN usermod -u $USER_ID $USER_NAME
WORKDIR /tmp/sciknowmap

ENV ES_PKG_NAME elasticsearch-2.3.3
RUN wget https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/tar/elasticsearch/2.3.3/$ES_PKG_NAME.tar.gz --quiet
RUN tar xvzf $ES_PKG_NAME.tar.gz
RUN rm -f $ES_PKG_NAME.tar.gz
RUN mv $ES_PKG_NAME /home/$USER_NAME/elasticsearch

RUN echo "root:iamgroot" | chpasswd
RUN echo "$USER_NAME:irobot" | chpasswd
RUN echo "path.data: /tmp/es_data/\ncluster.routing.allocation.disk.threshold_enabled: false" > /home/$USER_NAME/elasticsearch/config/elasticsearch.yml
RUN chown -R $USER_NAME /home/$USER_NAME/elasticsearch

COPY dot_history.sh /home/$USER_NAME/.bash_profile

# expose jupyter port 
EXPOSE 8888

# start jupyter at container start

CMD ["startup.sh"] 

COPY start-notebook.sh /usr/local/bin/ 
COPY start-elastic.sh /usr/local/bin/ 

# copy startup script into the container
COPY startup.sh /usr/local/bin/ 
