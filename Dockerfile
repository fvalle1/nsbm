FROM jupyter/datascience-notebook:latest

WORKDIR /home/jovyan/work
COPY environment.yml /home/jovyan/work.

RUN conda env create -f environment.yml
RUN conda activate trisbm
COPY run.py /home/jovyan/.

ENTRYPOINT ["python3", "/home/jovyan/run.py"]
CMD [""]
