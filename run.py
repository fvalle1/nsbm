import sys
sys.path.append("/home/jovyan/")
from trisbm import trisbm
import numpy as np
import os

os.chdir("/home/jovyan/work/")

model = trisbm()
model.load_graph("graph.xml.gz")    
    
model.fit(verbose=True)
    
state = model.state.copy(bs=model.state.get_bs() + [np.zeros(1)] * 4, sampling = True)

for _ in range(100):
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10, verbose=True)

print("Entropy gain :", 100*(model.state.entropy()-state.entropy())/max([model.state.entropy(),state.entropy()]))    

model.draw(output="network.pdf")

model.dump_model("model.pkl")