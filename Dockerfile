FROM jupyter/datascience-notebook:latest

WORKDIR /home/jovyan/work
COPY environment.yml /home/jovyan/work/.

RUN conda env create -f environment.yml
COPY run.py /home/jovyan/.

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "trisbm", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trisbm", "python3", "/home/jovyan/run.py"]
CMD [""]
