import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--train", action="store_true", help="Perform SISA training on the shard"
)
parser.add_argument("--test", action="store_true", help="Compute shard predictions")

parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Train for the specified number of epochs, default 20",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)
parser.add_argument(
    "--dropout_rate",
    default=0.4,
    type=float,
    help="Dropout rate, if relevant, default 0.4",
)
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001"
)

parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")

parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)

parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shard", type=int, help="Index of the shard to train/test")
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)
parser.add_argument(
    "--dataset",
    default="data/purchase/datasetfile",
    help="Location of the datasetfile, default data/purchase/datasetfile",
)

parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
)
parser.add_argument(
    "--label",
    default="latest",
    help="Label to be used on simlinks and outputs, default latest",
)
args = parser.parse_args()

if args.train and os.path.exists(f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"):
    exit()

import json
import torch
import numpy as np

from tqdm import tqdm
from glob import glob
from time import time
from importlib import import_module

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot
from torchmetrics import Accuracy

from sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchTestBatch, get_data_loader

# Import the architecture.

model_lib = import_module("architectures.{}".format(args.model))

# Use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member

# Retrive dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
nb_classes = datasetfile["nb_classes"]

# Instantiate model and send to selected device.
if hasattr(model_lib, 'Model'):
    input_shape = tuple(datasetfile["input_shape"])
    model = model_lib.Model(input_shape, nb_classes, dropout_rate=args.dropout_rate).to(device)
elif hasattr(model_lib, 'model'):
    model = model_lib.model.to(device)
    tokenizer = model_lib.tokenizer
else:
    raise "Unsupported model"

# Instantiate loss and optimizer.
loss_fn = CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
else:
    raise "Unsupported optimizer"

if args.train:
    shard_size = sizeOfShard(args.container, args.shard)
    slice_size = shard_size // args.slices
    avg_epochs_per_slice = (2 * args.slices / (args.slices + 1) * args.epochs / args.slices)
    loaded = False

    for sl in range(args.slices):
        # Get slice hash using sharded lib.
        slice_hash = getShardHash(args.container, args.label, args.shard, until=(sl + 1) * slice_size)


        # Initialize state.
        elapsed_time = 0
        start_epoch = 0
        slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)

        # If this is the first slice, no need to load anything.
        if sl == 0:
            loaded = True
        
        # If weights are already in memory (from previous slice), skip loading.
        if not loaded:
            # Look for a recovery checkpoint for the slice.
            recovery_list = glob(f"containers/{args.container}/cache/{slice_hash}_*.pt")
            if len(recovery_list) > 0:
                print(f"Recovery checkpoint found for shard {args.shard} on slice {sl}")

                # Load weights.
                model.load_state_dict(torch.load(recovery_list[-1]))
                start_epoch = int(recovery_list[-1].split("/")[-1].split(".")[0].split("_")[1])

                # Load time
                with open(f"containers/{args.container}/times/{slice_hash}_{start_epoch}.time","r") as f:
                    elapsed_time = float(f.read())

            # If there is no recovery checkpoint and this slice is not the first, load previous slice.
            elif sl > 0:
                previous_slice_hash = getShardHash(args.container, args.label, args.shard, until=sl * slice_size)
                # Load weights.
                model.load_state_dict(torch.load(f"containers/{args.container}/cache/{previous_slice_hash}.pt"))

            # Mark model as loaded for next slices.
            loaded = True

        # Actual training.
        accum_iter = 32 // args.batch_size if args.batch_size < 32 else 1

        train_time = 0.0
        for epoch in range(start_epoch, slice_epochs):
            epoch_start_time = time()

            if hasattr(model_lib, 'Model'):
                for inputs, labels in fetchShardBatch(
                    args.container,
                    args.label,
                    args.shard,
                    args.batch_size,
                    args.dataset,
                    until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
                ):

                    # Convert data to torch format and send to selected device.
                    gpu_inputs = torch.from_numpy(inputs).to(device)  # pylint: disable=no-member
                    gpu_labels = torch.from_numpy(labels).to(device)  # pylint: disable=no-member

                    forward_start_time = time()

                    # Perform basic training step.
                    logits = model(gpu_inputs)
                    loss = loss_fn(logits, gpu_labels)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    train_time += time() - forward_start_time
            else:
                dataloader = get_data_loader(
                    args.container,
                    args.label,
                    args.shard,
                    args.batch_size,
                    tokenizer,
                    args.dataset,
                    until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
                )

                with tqdm(dataloader) as pbar:
                    for batch_idx, inputs in enumerate(pbar):
                        forward_start_time = time()

                        # Perform basic training step.
                        logits = model(**dict(inputs.to(device)))
                        loss = logits.loss / accum_iter
                        loss.backward()

                        if (batch_idx + 1) % accum_iter == 0 or batch_idx + 1 == len(dataloader):
                            optimizer.step()
                            optimizer.zero_grad()

                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                        train_time += time() - forward_start_time

            # Create a checkpoint every chkpt_interval.
            if (args.chkpt_interval != -1 and epoch % args.chkpt_interval == args.chkpt_interval - 1):
                # Save weights
                torch.save(model.state_dict(), f"containers/{args.container}/cache/{slice_hash}_{epoch}.pt")

                # Save time
                with open(f"containers/{args.container}/times/{slice_hash}_{epoch}.time","w") as f:
                    f.write("{}\n".format(train_time + elapsed_time))

        else:
            # When training is complete, save slice.
            torch.save(model.state_dict(), f"containers/{args.container}/cache/{slice_hash}.pt")
            with open(f"containers/{args.container}/times/{slice_hash}.time", "w") as f:
                f.write(f"{train_time + elapsed_time}\n")

            # Remove previous checkpoint.
            for previous_chkpt in glob(f"containers/{args.container}/cache/{slice_hash}_*.pt"):
                os.remove(previous_chkpt)
            for previous_chkpt in glob(f"containers/{args.container}/times/{slice_hash}_*.time"):
                os.remove(previous_chkpt)
    else:
        # If this is the last slice, create a symlink attached to it.
        os.symlink(f"{slice_hash}.pt",
            f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt")
        os.symlink(f"{slice_hash}.time",
            f"containers/{args.container}/times/shard-{args.shard}:{args.label}.time")
        


if args.test:
    # Load model weights from shard checkpoint (last slice).
    model.load_state_dict(torch.load(f"containers/{args.conatainer}/cache/shard-{args.shard}:{args.label}.pt"))

    # Compute predictions batch per batch.
    outputs = np.empty((0, nb_classes))
    for inputs, _ in fetchTestBatch(args.dataset, args.batch_size):
        # Convert data to torch format and send to selected device.
        gpu_inputs = torch.from_numpy(inputs).to(device)  # pylint: disable=no-member

        if args.output_type == "softmax":
            # Actual batch prediction.
            logits = model(gpu_inputs)
            predictions = torch.softmax(logits, dim=1).to("cpu")  # Send back to cpu.
        
            # Convert back to numpy and concatenate with previous batches.
            outputs = np.concatenate((outputs, predictions.numpy()))

        else:
            # Actual batch prediction.
            logits = model(gpu_inputs)
            predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

            # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
            out = one_hot(predictions, nb_classes).to("cpu")
            outputs = np.concatenate((outputs, out.numpy()))

    # Save outputs in numpy format.
    outputs = np.array(outputs)
    np.save(f"containers/{args.container}/outputs/shard-{args.shard}:{args.label}.npy", outputs)
