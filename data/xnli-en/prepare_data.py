import json
import torch
from datasets import load_dataset 

if __name__ == '__main__':

    dataset = load_dataset("xnli", 'en')

    with open('datasetfile', mode='w') as f:
        f.write(json.dumps({
            "nb_train": len(dataset['train']),
            "nb_test": len(dataset['validation']),
            "input_shape": 512, # max token length
            "nb_classes": len(set(dataset['train']['label'])),
            "dataloader": "dataloader"
        }, indent=4))
