# FlowGraph_SoftwareSecurity

### Environment Create 


### Data
* The preprocessed Cooking (COR), Maintenance Manual (MAM) and our created dataset WHPG data are available in the ```data/``` directory
* For statistics of the pre-processed data, please refer to our paper
* Each folder has the train.jsonl, test.jsonl, and val.jsonl used for our experiments

### Data Process
* Run ```python data2format.py```, aiming to transfer the init data into jsonl format. 
* Run ```python AddSynInfo.py```, aiming to add the syntactic information into the data files. 
* 8 files (i.e., link2id.json, linkType2id.json, pos2id.json, syn_rel2id.json, test.jsonl, train.jsonl, valid.jsonl) are obtained 


### Training Model 
#### Hyper-Parameters
* window_size: 0 : refer to the model does not have the window limitation. 
* model = bert-base-uncased /  roberta-base 

### Run  
* train the main model 
```
CUDA_VISIBLE_DEVICES=3 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 0 --model bert-base-uncased --datafile data/WHPG/ > ./log/ALL_wikiHow_log.txt
```




