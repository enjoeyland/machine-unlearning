import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BatchEncoding

class XNLIEnDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

        self.encodings = self.tokenizer(
            data['premise'], data['hypothesis'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        self.encodings['labels'] = torch.tensor(self.data['label'])
        self.num_classes = len(set(self.data['label']))

    def __getitem__(self, idx):
        item  = {key: val[idx].squeeze() for key, val in self.encodings.items()}
        return BatchEncoding(data=item, tensor_type='pt')

    def __len__(self):
        return len(self.data)

dataset = load_dataset("xnli", 'en')

def get_dataset(tokenizer, max_length, category='train'):
    return XNLIEnDataset(tokenizer, dataset[category], max_length)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    checkpoint = "google/mt5-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print("Tokenizing...", end="")
    eval_dataset = get_dataset(tokenizer, category='test')
    print("Done")


    print(dict(eval_dataset[[0,2,3]].to("cuda")))



    # print(eval_dataset['label'][[0,2,3]])

# def load(indices, category='train'):
#     indices = indices.astype(int)
#     if category == 'train':
#         return train_dataset[indices], y_train[indices]
#     elif category == 'test':
#         return eval_dataset[indices], y_test[indices]

# pwd = os.path.dirname(os.path.realpath(__file__))

# train_data = np.load(os.path.join(pwd, 'purchase2_train.npy'), allow_pickle=True)
# test_data = np.load(os.path.join(pwd, 'purchase2_test.npy'), allow_pickle=True)

# train_data = train_data.reshape((1,))[0]
# test_data = test_data.reshape((1,))[0]

# X_train = train_data['X'].astype(np.float32)
# X_test = test_data['X'].astype(np.float32)
# y_train = train_data['y'].astype(np.int64)
# y_test = test_data['y'].astype(np.int64)

# def load(indices, category='train'):
#     indices = indices.astype(int)
#     if category == 'train':
#         return X_train[indices], y_train[indices]
#     elif category == 'test':
#         return X_test[indices], y_test[indices]