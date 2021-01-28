FROM jupyter/datascience-notebook

RUN conda install -c conda-forge mamba -y
RUN mamba install graph-tool -y


WORKDIR /home/jovyan/work
COPY run.py /home/jovyan/.
COPY trisbm.py /home/jovyan/.

ENTRYPOINT ["python3", "/home/jovyan/run.py"]
CMD [""]
