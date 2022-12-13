FROM python:3.10.4

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

#Create non-root user
RUN useradd -ms /bin/bash python-api
USER python-api

#Copy the source files into the container
WORKDIR /repo
COPY . /repo

#Install pip requirements
RUN python -m pip install -r requirements.txt

#Define the ommand to be run when the container is started 
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["imdb_score.py"]