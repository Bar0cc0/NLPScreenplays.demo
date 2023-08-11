FROM python:slim-bullseye

# set user permission
RUN apt update \
    && apt -y install sudo \
    && useradd -m root | chpasswd \
    && adduser root sudo
USER root

# install firefox
RUN apt install tar \
    && apt install bzip2 \
    && apt -y install wget \
    && sudo apt-get -y install libgtk-3-0 libgtk-3-common libasound2 libdbus-glib-1-2 libx11-xcb1 --fix-missing \
    && mkdir /usr/lib/firefox \
    && wget "https://download.cdn.mozilla.net/pub/firefox/releases/117.0b5/linux-x86_64/en-US/firefox-117.0b5.tar.bz2" -P /usr/lib/firefox \
    && tar -xjf /usr/lib/firefox/firefox-117.0b5.tar.bz2 -C /usr/lib/firefox \
    && ln -s /usr/lib/firefox/firefox/firefox /usr/bin/firefox

# redirect to host
ENV DISPLAY=host.docker.internal:0.0

# install airflow
RUN pip install apache-airflow==2.6.3

# install postgres
RUN apt -y install postgresql

# set marto env
WORKDIR /marto
COPY ./ ./
RUN pip install --no-cache-dir -r requirements.txt 

# entrypoint
CMD [ "bash", "startup.sh" ]
EXPOSE 8080