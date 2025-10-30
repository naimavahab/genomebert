import matplotlib.pyplot as plt
import numpy as np

# Datasets and MCC values
data3 = {

    "H3K9ac": [0.54, 0.45, 0.41, 0.537120236],
    "H4ac": [0.54, 0.52, 0.31, 0.490953763],
"H3K4me2": [0.31, 0.28, 0.23, 0.298709859],
"H3K4me3": [0.52, 0.44, 0.19, 0.362646774],
 "H3K79me3": [0.63, 0.62, 0.54, 0.607123437]
 
   
   
}

data = {"splice_sites_all": [0.9, 0.87, 0.28, 0.973053033],
 "H4": [0.79, 0.8, 0.71, 0.794327364],  
 "enhancers_types": [0.42, 0.42, 0.54, 0.406288417],
    "promoter_all": [0.909, 0.94, 0.92, 0.927072113],     
    "H3": [0.76, 0.819, 0.69, 0.791204681],    
     }

data1 = {    
    "H3K14ac": [0.52, 0.53, 0.33, 0.514706079],    
    "H3K4me1": [0.5, 0.52, 0.32, 0.517027317]   ,
     "H3K36me3": [0.63, 0.64, 0.41, 0.601377336] }

data = {    
    "splice site donor": [0.82, 0.77, 0.52, 0.92],    
    "splice site acceptor": [0.84, 0.82, 0.76, 0.24]   ,
     "ncrna family bnoise0": [0.97, 0.98, 0.84, 0.94],
     "ncrna family bnoise200": [0.95, 0.96, 0.24, 0.95]}
models = ["genomicBERT", "DNABERT2", "GenaLM", "NT"]
colors = ['#2ca02c', '#F0B090', '#DFE092', '#E0B4C1']

datasets = list(data.keys())
values = np.array(list(data.values()))  # shape: (num_datasets, num_models)

x = np.arange(len(datasets))  # dataset positions
width = 0.2  # bar width

plt.figure(figsize=(14,6))

# plot each model group
for i, (model, color) in enumerate(zip(models, colors)):
    plt.bar(x + i*width, values[:, i], width, label=model, color=color)

# labels and formatting
plt.xticks(x + width*1.5, datasets, rotation=45, ha="right")
plt.ylabel("MCC")
plt.title("MCC Comparison Across Datasets and Models")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()

plt.savefig('mccrna.png')