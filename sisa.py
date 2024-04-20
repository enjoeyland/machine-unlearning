import os
import json
import torch
import wandb
import numpy as np
import argparse
import torchmetrics

from tqdm import tqdm
from glob import glob
from time import time
from copy import deepcopy
from importlib import import_module

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam, SGD
from torch.nn.functional import one_hot

from sharded import sizeOfShard, shard_dataloader, eval_dataloader
from transformers import set_seed

def add_arguments(parser):
    # Model arguments
    parser.add_argument("--model", default="purchase", help="Architecture to use, default purchase")
  
    # Data arguments
    parser.add_argument("--dataset", default="data/purchase/datasetfile", help="Location of the datasetfile, default data/purchase/datasetfile")
    
    parser.add_argument("--container", help="Name of the container")
    parser.add_argument("--shard", type=int, help="Index of the shard to train/test")
    parser.add_argument("--label", default="latest", help="Label to be used on simlinks and outputs, default latest")
    
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # Training arguments
    parser.add_argument("--train", action="store_true", help="Perform SISA training on the shard")
    parser.add_argument("--test", action="store_true", help="Compute shard predictions")

    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size, default 16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--save_strategies", default="epoch", help="Save strategies, default epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluation_steps", type=int, default=50)
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training, default False")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory, default False")

    parser.add_argument("--epochs", default=20, type=int, help="Train for the specified number of epochs, default 20")
    parser.add_argument("--slices", default=1, type=int, help="Number of slices to use, default 1")

    parser.add_argument("--bf16_full_eval", action="store_true", help="Use full evaluation in bf16, default False")
    parser.add_argument("--output_type", default="argmax", help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax")


def train(args):
    device = args.device
    set_seed(args.seed)

    # Import the architecture.
    model_lib = import_module("architectures.{}".format(args.model))

    # Instantiate model and send to selected device.
    if hasattr(model_lib, 'model'):
        model = model_lib.model.to(device)
        tokenizer = model_lib.tokenizer
    else:
        raise "Unsupported model"

    train_dataset = args.dataloader_module.get_dataset(tokenizer, max_length=args.max_length, category='train')
    eval_dataset = args.dataloader_module.get_dataset(tokenizer, max_length=args.max_length, category='validation')

    # Instantiate loss and optimizer.
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise "Unsupported optimizer"

    shard_size = sizeOfShard(args.container, args.shard)
    slice_size = shard_size // args.slices
    avg_epochs_per_slice = (2 * args.epochs / (args.slices + 1)) # See paper for explanation.
    loaded = False
    best_model_state = None
    train_state = {"loss": 0.0, "eval_accuracy": 0.0, "eval_loss": 0.0, "slice": 0, "model_step": 0, "step": 0, "time": 0.0}
    for sl in range(args.slices):
        train_state["slice"] = sl
        print(f"slice {sl+1}/{args.slices}")
        # Get slice hash using sharded lib.
        slice_name = f"shard{args.shard}_label{args.label}_slice{sl}_until{(sl + 1) * slice_size}"

        if os.path.exists(f"containers/{args.container}/cache/{slice_name}.pt"):
            continue

        # Initialize state.
        start_epoch = 0
        slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)
        
        # If weights are already in memory (from previous slice), skip loading.
        if not loaded:
            # Look for a recovery checkpoint for the slice.
            recovery_list = glob(f"containers/{args.container}/cache/{slice_name}_epoch*.pt")
            if len(recovery_list) > 0:
                print(f"Recovery checkpoint found for shard {args.shard} on slice {sl}")
                start_epoch = int(recovery_list[-1].split("/")[-1].split(".")[0].split("_epoch")[1])
                model.load_state_dict(torch.load(f"containers/{args.container}/cache/{slice_name}_epoch{start_epoch}.pt"))
                best_model_state = deepcopy(model.state_dict())
                for k, v in best_model_state.items():
                    best_model_state[k] = v.cpu()
                train_state = json.load(open(f"containers/{args.container}/cache/{slice_name}_epoch{start_epoch}.json", "r"))

            # If there is no recovery checkpoint and this slice is not the first, load previous slice.
            elif sl > 0:
                previous_slice_name = f"shard{args.shard}_label{args.label}_slice{sl-1}_until{sl * slice_size}"
                model.load_state_dict(torch.load(f"containers/{args.container}/cache/{previous_slice_name}.pt"))
                best_model_state = deepcopy(model.state_dict())
                for k, v in best_model_state.items():
                    best_model_state[k] = v.cpu()
                train_state = json.load(open(f"containers/{args.container}/cache/{previous_slice_name}.json", "r"))

            # Mark model as loaded for next slices.
            loaded = True

        for epoch in range(start_epoch, slice_epochs):
            dataloader = shard_dataloader(
                args.container,
                args.label,
                args.shard,
                args.batch_size,
                train_dataset,
                until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
            )
            with tqdm(dataloader) as pbar:
                for batch_idx, inputs in enumerate(pbar):
                    model.train()
                    model.float()
                    train_state["step"] += 1
                    forward_start_time = time()

                    with torch.autocast(device_type=args.device.type, dtype=torch.bfloat16, enabled=args.bf16):
                        outputs = model(**dict(inputs.to(device)))
                        loss = outputs.loss / args.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx + 1 == len(dataloader):
                        optimizer.step()
                        optimizer.zero_grad()

                    train_state["time"] += time() - forward_start_time
                    pbar.set_postfix({"loss": outputs.loss.item()})

                    if args.logging_steps > 0 and (batch_idx + 1) % args.logging_steps == 0:
                        train_state["loss"] = outputs.loss.item()
                        wandb.log({"train":{"loss": outputs.loss.item(), "slice": sl}}, step=train_state["step"])

                    if args.evaluation_steps > 0 and (batch_idx + 1) % args.evaluation_steps == 0:
                        results = test(args, model=model, dataset=eval_dataset)
                        wandb.log({"eval": results}, step=train_state["step"])
                        print(f"Step {train_state['step']}: {results}")

                        if args.load_best_model_at_end and results["accuracy"] > train_state["eval_accuracy"]:
                            best_model_state = deepcopy(model.state_dict())
                            for k, v in best_model_state.items():
                                best_model_state[k] = v.cpu()
                            train_state["eval_accuracy"] = results["accuracy"]
                            train_state["eval_loss"] = results["loss"]
                            train_state["model_step"] = train_state["step"]
               
                else:
                    # last epoch
                    wandb.log({"train":{"loss": outputs.loss.item(), "slice": sl}}, step=train_state["step"])

                    results = test(args, model=model, tokenizer=tokenizer)
                    wandb.log({"eval": results}, step=train_state["step"])

                    if args.load_best_model_at_end and results["accuracy"] > train_state["eval_accuracy"]:
                        best_model_state = deepcopy(model.state_dict())
                        for k, v in best_model_state.items():
                            best_model_state[k] = v.cpu()
                        train_state["eval_accuracy"] = results["accuracy"]
                        train_state["eval_loss"] = results["loss"]
                        train_state["model_step"] = train_state["step"]
                
                    if args.load_best_model_at_end:
                        for k, v in best_model_state.items():
                            best_model_state[k] = v.to(device)
                        torch.save(best_model_state, f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.pt")
                        for k, v in best_model_state.items():
                            best_model_state[k] = v.cpu()
                        json.dump(train_state, open(f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.json", "w"))
                    else:
                        torch.save(model.state_dict(), f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.pt")
                        train_state["eval_accuracy"] = results["accuracy"]
                        train_state["eval_loss"] = results["loss"]
                        train_state["model_step"] = train_state["step"]
                        json.dump(train_state, open(f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.json", "w"))
                

        else:
            # last slice
            if args.save_strategies != "no":
                # When training is complete, save slice.
                os.rename(f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.pt",
                        f"containers/{args.container}/cache/{slice_name}.pt")
                os.rename(f"containers/{args.container}/cache/{slice_name}_epoch{epoch}.json",
                    f"containers/{args.container}/cache/{slice_name}.json")

                # Remove previous checkpoint.
                for previous_chkpt in glob(f"containers/{args.container}/cache/{slice_name}_epoch*.pt"):
                    os.remove(previous_chkpt)

    else:
        # If this is the last slice, create a symlink attached to it.
        os.symlink(f"{slice_name}.pt",
            f"containers/{args.container}/cache/shard{args.shard}_label{args.label}.pt")
        os.symlink(f"{slice_name}.json",
            f"containers/{args.container}/cache/shard{args.shard}_label{args.label}.json")


def test(args, model=None, dataset=None):
    num_classes = args.num_classes
    device = args.device

    save = False
    if model is None:
        save = True
        # Import the architecture.
        model_lib = import_module("architectures.{}".format(args.model))

        # Instantiate model and send to selected device.
        if hasattr(model_lib, 'model'):
            # Load model weights from shard checkpoint (last slice).
            model = model_lib.model
            model.load_state_dict(torch.load(f"containers/{args.container}/cache/shard{args.shard}_label{args.label}.pt"))
            model.to(device)
        else:
            raise "Unsupported model"

    if dataset is None:
        model_lib = import_module("architectures.{}".format(args.model))
        if hasattr(model_lib, 'model'):
            tokenizer = model_lib.tokenizer
            dataset = args.dataloader_module.get_dataset(tokenizer, max_length=args.max_length, category='validation')
        else:
            raise "Unsupported model"

    model.eval()
    if args.bf16_full_eval:
        model.bfloat16()
    
    accuracy_score = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    all_outputs = torch.empty(0, num_classes)
    predictions = torch.tensor([])
    references = torch.tensor([])
    loss = 0
    dataloader = eval_dataloader(
        args.batch_size,
        dataset,
    )
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader)):
            outputs = model(**dict(inputs.to(device)))
            logits = outputs.logits

            loss += CrossEntropyLoss()(logits, inputs['labels'])
            preds = torch.argmax(logits, dim=-1)
            predictions = torch.cat((predictions, preds.cpu()))
            references = torch.cat((references, inputs['labels'].cpu()))

            if args.output_type == "argmax":
                all_outputs = torch.cat((all_outputs, one_hot(preds, num_classes).cpu()))
            elif args.output_type == "softmax":
                all_outputs = torch.cat((all_outputs, torch.softmax(logits, dim=1).cpu()))
            
        loss /= len(dataloader)
        accuracy = accuracy_score(predictions, references)
    
    if save:
        np.save(f"containers/{args.container}/outputs/shard{args.shard}_label{args.label}.npy", all_outputs.numpy())
    
    return {"loss": loss.item(), "accuracy": accuracy.item()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    assert 2 * args.epochs >= args.slices + 1, "Not enough epochs per slice"

    if args.train and not args.overwrite_output_dir \
            and os.path.exists(f"containers/{args.container}/cache/shard{args.shard}_label{args.label}.pt"):
        exit()

    if args.train:
        wandb.init(project="SISA", config=args, name=f"{args.container}-{args.shard}")

    with open(args.dataset) as f:
        datasetfile = json.loads(f.read())
    args.dataloader_module = import_module('.'.join(args.dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    args.num_classes = datasetfile["nb_classes"]

    # Use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
    args.device = device

    if args.train:
        train(args)

    if args.test:
        results = test(args)
        print(results)
    