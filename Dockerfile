FROM jupyter/datascience-notebook

RUN conda install -c conda-forge mamba -y
RUN mamba install graph-tool -y


WORKDIR /home/jovyan/work
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
CMD [""]
