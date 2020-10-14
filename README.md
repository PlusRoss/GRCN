Code for [Graph-Revised Convolutional Network](https://arxiv.org/abs/1911.07123) ([ECML-PKDD 2020](https://ecmlpkdd2020.net/))  

## Requirements
```
python >= 3.6.0
pytorch = 1.5.0
tqdm
itermplot
```
The code is based on [pyg](https://github.com/rusty1s/pytorch_geometric). Please see [instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for its installation.

*dataprocess.py* is used for data spliting, edge sampling, and data loader.   


## Reproduce Results
### Run our model GRCN under fixed train/val/test split
```
./run_fixed.sh 1(GPU No.) GRCN Cora(dataset: Cora, CiteSeer, PubMed) --sparse
```   
To save the log result, add `--save` in the command.  
You can change the parameters of *run_fixed.sh* and *config/*.    


### Run our model GRCN under random train/val/test split
```
./run_random.sh 1(GPU No.) GRCN Cora(dataset: Cora, CiteSeer, PubMed, CoraFull, Computers, CS) --sparse
```   
When running on PubMed dataset, add `--keep_train_num`.  
To save the log result, add `--save` in the command.  
You can change the parameters of *run_random.sh* and *config/*.    

### Results

Our model achieves the following performance on :

#### semi-supervised node classification ([fixed split](https://arxiv.org/pdf/1603.08861.pdf))

| Model     | Cora     | CiteSeer | PubMed   |
|-----------|----------|----------|----------|
| GCN       | 81.4±0.5 | 70.9±0.5 | 79.0±0.3 |
| GAT       | 83.2±0.7 | 72.6±0.6 | 78.8±0.3 |
| LDS       | 84.0±0.4 | 74.8±0.5 | N/A      |
| GLCN      | 81.8±0.6 | 70.8±0.5 | 78.8±0.4 |
| Fast-GRCN | 83.6±0.4 | 72.9±0.6 | 79.0±0.2 |
| GRCN      | 84.2±0.4 | 73.6±0.5 | 79.0±0.2 |

#### semi-supervised node classification (random splits)   

| Model     | Cora     | CiteSeer | PubMed   |
|-----------|----------|----------|----------|
| GCN       | 81.2±1.9 | 69.8±1.9 | 77.7±2.9 |
| GAT       | 81.7±1.9 | 68.8±1.8 | 77.7±3.2 |
| LDS       | 81.6±1.0 | 71.0±0.9 | N/A      |
| GLCN      | 81.4±1.9 | 69.8±1.8 | 77.2±3.2 |
| Fast-GRCN | 83.8±1.6 | 72.3±1.4 | 77.6±3.2 |
| GRCN      | 83.7±1.7 | 72.6±1.3 | 77.9±0.2 |
