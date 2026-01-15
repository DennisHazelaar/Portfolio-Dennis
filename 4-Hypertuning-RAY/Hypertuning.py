from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import Trainer, TrainerSettings, metrics, ReportTypes
from mltrainer.preprocessors import BasePreprocessor

from models.model import CNN

import mlflow


# -------------------------
# MLflow setup
# -------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("cnn_flowers_capacity_v2")


# -------------------------
# Constants
# -------------------------
MAX_EPOCHS = 5
BATCH_SIZE = 32


# -------------------------
# Dataset
# -------------------------
def get_flowers_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    preprocessor = BasePreprocessor()

    streamers = factory.create_datastreamer(
        batchsize=batchsize,
        preprocessor=preprocessor,
    )

    trainstreamer = streamers["train"].stream()
    validstreamer = streamers["valid"].stream()
    return trainstreamer, validstreamer


# -------------------------
# Single experiment
# -------------------------
def run_experiment(
    filters: int,
    units1: int,
    units2: int,
    lr: float,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainstreamer, validstreamer = get_flowers_streamers(batch_size)

    model = CNN(
        filters=filters,
        units1=units1,
        units2=units2,
        input_size=(batch_size, 3, 64, 64),
        n_classes=102,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()

    settings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy],
        logdir=Path("logs"),
        train_steps=50,
        valid_steps=25,
        reporttypes=[ReportTypes.MLFLOW],
        optimizer_kwargs={"lr": lr},
)


    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,      
        traindataloader=trainstreamer,
        validdataloader=validstreamer,
        scheduler=None,
        device=device,
)

    mlflow.set_tags({
        "filters": filters,
        "units1": units1,
        "units2": units2,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    })

    trainer.loop()
    mlflow.end_run()

# -------------------------
# Experiment loop (GRID SEARCH)
# -------------------------
def main():

    filters_list = [32]
    units1_list = [64, 256, 512]
    units2_list = [64, 256, 512]
    lrs = [1e-3]

    for filters in filters_list:
        for units1 in units1_list:
            for units2 in units2_list:
                for lr in lrs:
                    print(
                        f"Running experiment: "
                        f"filters={filters}, units1={units1}, "
                        f"units2={units2}, lr={lr}"
                    )

                    run_experiment(
                        filters=filters,
                        units1=units1,
                        units2=units2,
                        lr=lr,
                    )


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
