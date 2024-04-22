import argparse
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models import MutlilinugalModel



def add_arguments(parser):
    # Model arguments
    parser.add_argument("--model", default="purchase", help="Architecture to use, default purchase")
  
    # Data arguments
    parser.add_argument("--dataset", default="data/purchase/datasetfile", help="Location of the datasetfile, default data/purchase/datasetfile")
    
    parser.add_argument("--name", help="Name of the model")
    parser.add_argument("--container", help="Name of the container")
    
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

    parser.add_argument("--epochs", default=20, type=int, help="Train for the specified number of epochs, default 20")

    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--evaluation_strategy", default="steps", choices=["steps", "epoch", "no"], help="Evaluation strategy, default steps")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--bf16_full_eval", action="store_true", help="Use full evaluation in bf16, default False")
    parser.add_argument("--output_type", default="argmax", help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax")
    
    parser.add_argument("--save_strategies", default="epoch", choices=["epoch", "no"], help="Save strategies, default epoch")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training, default False")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory, default False")

def main(args):
    L.seed_everything(args.seed, workers=True)

    wandb_logger = WandbLogger(project="SISA", config=args, name=f"{args.name}")
    model = MutlilinugalModel(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        filename="{epoch}-{global_step}-{val_accuracy:.4f}",
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
    )

    trainer = L.Trainer(
        default_root_dir=f"containers/{args.container}/cache",
        devices="auto",
        precision="bf16-mixed" if args.bf16 else "32-true",
        max_epochs=args.epochs,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=args.logging_steps,
        val_check_interval=args.eval_steps,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    main(args)    