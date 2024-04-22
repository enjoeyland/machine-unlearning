import json
import lightning as L

from importlib import import_module

from torch.optim import AdamW, Adam, SGD

from sharded import eval_dataloader, train_dataloader


class MutlilinugalModel(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        model_lib = import_module("architectures.{}".format(hparams.model))
        self.model = model_lib.model
        self.tokenizer = model_lib.tokenizer

        with open(hparams.dataset) as f:
            datasetfile = json.loads(f.read())
        self.dataloader_module = import_module('.'.join(hparams.dataset.split('/')[:-1] + [datasetfile['dataloader']]))
        self.num_classes = datasetfile["nb_classes"]

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**dict(batch))
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**dict(batch))
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"val_loss": loss, "val_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**dict(batch))
        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy}, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        # schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        # return {"optimizer": optimizer, "lr_schedular": {"scheduler": schedular, "interval": "step"}}
        return {"optimizer": optimizer}

    def train_dataloader(self):
        dataset = self.dataloader_module.get_dataset(self.tokenizer, max_length=self.hparams.max_length, category='train')
        return train_dataloader(self.hparams.batch_size, dataset)
    
    def val_dataloader(self):
        dataset = self.dataloader_module.get_dataset(self.tokenizer, max_length=self.hparams.max_length, category='validation')
        return eval_dataloader(self.hparams.batch_size, dataset)

    def test_dataloader(self):
        dataset = self.dataloader_module.get_dataset(self.tokenizer, max_length=self.hparams.max_length, category='test')
        return eval_dataloader(self.hparams.batch_size, dataset)