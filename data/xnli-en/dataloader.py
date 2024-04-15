import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BatchEncoding

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=256):
        self.encodings = tokenizer(data['premise'], data['hypothesis'], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        start_token_id = tokenizer.pad_token_id
        self.labels = self.encodings.input_ids.clone()
        decoder_input_ids = torch.full_like(self.labels, start_token_id)
        decoder_input_ids[:, 1:] = self.labels[:, :-1].clone()
        self.encodings['labels'] = self.labels
        self.encodings['decoder_input_ids'] = decoder_input_ids

    def __getitem__(self, idx):
        item  = {key: val[idx].squeeze() for key, val in self.encodings.items()}
        return BatchEncoding(data=item, tensor_type='pt')

    def __len__(self):
        return len(self.encodings.input_ids)


def get_dataloader(tokenizer, max_length=256, category='train'):
    category = 'validation' if category == 'test' else category
    dataset = load_dataset("xnli", 'en')
    return XNLIDataset(tokenizer, dataset[category], max_length)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    checkpoint = "google/mt5-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print("Tokenizing...", end="")
    eval_dataset = get_dataloader(tokenizer, category='test')
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