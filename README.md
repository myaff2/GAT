# GAT-ASD
The code for constructing the entire graphs and the model code have been open-sourced. 

## Dependency
We used python=3.7, pytorch=1.13.1, and torch-geometric=2.3.1 in our experiments.

## Data Preparation
1) Download the FisheyeMeeting audio-visual features, the annotation csv files and the SOTA model from [Google Drive](https://drive.google.com/file/d/1qlnYE22iqaWD9UKwV1JrF3SRi0HxepBn/view?usp=drive_link). The directories should look like as follows:
```
|-- features
    |-- FisheyeMeeting
        |-- train_forward
        |-- val_forward
|-- csv_files_FisheyeMeeting
    |-- FM_train_orig.csv
    |-- FM_train_val.csv
|-- best_for_FM.pt
```
## Graphs Construction
Run `generate_graph.py` to create the entire graphs from the features, you can modify the graph construction's hyperparameters '--numv','--time_edge'，'--short_time_edge':
```
python generate_graph.py 
```
## Train：
After the graph is constructed, you can use the generated graph for model training. Use `train_val.py` to train the model:
```
python train_val.py 
```
## Evaluate：
you can perform evaluation with the best model using the command：
```
python train_val.py --evaluation
```
### Acknowledgments
Thanks for the support of Min, Kyle's open source [repository](https://github.com/SRA2/SPELL) for this research.
