FROM jupyter/datascience-notebook:latest

RUN conda install -c conda-forge mamba -y
RUN mamba install graph-tool -y

WORKDIR /home/jovyan/work
COPY run.py /home/jovyan/.
COPY trisbm.py /home/jovyan/.
COPY requirements.txt /home/jovyan/.

RUN python3 -m pip install -r /home/jovyan/requirements.txt

ENTRYPOINT ["python3", "/home/jovyan/run.py"]
CMD [""]
