import os
import shutil
import pandas as pd
import numpy as np

def copy_and_rename_file(src_path, dest_path, new_filename):
    try:
        # Copy the file
        shutil.copy2(src_path, dest_path)
        
        # Rename the copied file
        new_path = f"{dest_path}/{new_filename}"
        shutil.move(f"{dest_path}/{src_path.split('/')[-1]}", new_path)
        
        print(f"File copied and renamed to: {new_path}")
    except Exception as e:
        print(f"Error: {e}")


cols = ["Discriminator HP", "Generator HP", "Number of traces to reach GE=1"]
results_root_path = "C:/Users/Sengim/Datasets/model_search_best"
df = pd.DataFrame(columns=cols)
for subdir in os.listdir(results_root_path):
    #print(subdir)
    if  not subdir.__contains__("simulat"):
        temp = subdir.split("vs_")
        hp = np.load(results_root_path + "/"+ subdir + "/hp.npz", allow_pickle=True)
        hpd = hp["hp_d"][()]
        del hpd["neurons_bilinear"]
        del hpd["layers_bilinear"]
        hpg = hp["hp_g"]

        nt = np.load(results_root_path + "/"+ subdir + "/metrics.npz", allow_pickle=True)['nt_fake'][0]


        index = temp[1]

    
        nt = int(input(f"Enter value for {index}: "))
        index = str(input(f"Enter name for {index}: "))
        print(nt)
        if nt >= 2000:
            continue
        data = {"Discriminator HP":hpd, "Generator HP":hpg, "Number of traces to reach GE=1":nt}
        df = df._append(pd.Series(data, name=index))

df.to_pickle("lda_pca_architectures.pkl")
