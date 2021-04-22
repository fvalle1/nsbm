import sys
sys.path.append("/home/jovyan/")
from trisbm import trisbm
import numpy as np
import pandas as pd
import os

os.chdir("/home/jovyan/work/")

model = trisbm()
if "graph.xml.gz" in os.listdir():
    model.load_graph("graph.xml.gz")
else:
    df = pd.DataFrame(
    index = ["w{}".format(w) for w in range(1000)],
    columns = ["doc{}".format(w) for w in range(250)],
    data = np.random.randint(1, 100, 250000).reshape((1000, 250)))

    df_key_list = [
        pd.DataFrame(
    index = ["w{}".format(w) for w in range(100+ik)],
    columns = ["doc{}".format(w) for w in range(250)],
    data = np.random.randint(1, 5+ik, (100+ik)*250).reshape((100+ik, 250)))
        
        for ik in range(3)
    ]
    model.make_graph_multiple_df(df, df_key_list)
    
model.fit(verbose=True)
    
# state = model.state.copy(bs=model.state.get_bs() + [np.zeros(1)] * 4, sampling = True)

# for _ in range(100):
#     state.multiflip_mcmc_sweep(beta=np.inf, niter=10, verbose=True)

# print("Entropy gain :", 100*(model.state.entropy()-state.entropy())/max([model.state.entropy(),state.entropy()]))    

model.dump_model("model.pkl")

model.save_data()

model.draw(output="network.pdf")
