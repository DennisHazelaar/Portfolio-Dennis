from datetime import datetime
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device


# There are more models in mltrainer.imagemodels for inspiration.
# You can import them, or create your own like here.
class CNN(nn.Module):
    def __init__(
        self,
        filters,
        units1,
        units2,
        n_conv_blocks,
        dropout=0.0,
        input_size=(32, 1, 28, 28),
    ):
        super().__init__()

        self.input_size = input_size
        in_channels = input_size[1]

        layers = []

        for i in range(n_conv_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else filters,
                    filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            if i < 4:
                layers.append(nn.MaxPool2d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.convolutions = nn.Sequential(*layers)

        activation_map_size = self._conv_test()
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, 10),
        )

    def _conv_test(self):
        x = torch.ones(self.input_size)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        return self.dense(x)


def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_path)


def objective(params):
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created {modeldir}")
    batchsize = 64
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=3,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.MLFLOW],
    )
    # Start a new MLflow run for tracking the experiment
    device = get_device()
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "Dennis")
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("n_conv_blocks", params["n_conv_blocks"])
        mlflow.log_param("dropout", params["dropout"])

        mlflow.log_param("batchsize", f"{batchsize}")

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate the CNN model with the given hyperparameters
        model = CNN(**params)
        model.to(device)
        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        return {"loss": trainer.test_loss, "status": STATUS_OK}


def main():
    setup_mlflow("mlflow_database")

    depths = [2, 4, 6]
    dropouts = [0.0, 0.3, 0.5]

    for depth in depths:
        for dropout in dropouts:
            params = {
                "filters": 32,
                "units1": 128,
                "units2": 64,
                "n_conv_blocks": depth,
                "dropout": dropout,
            }

            logger.info(
                f"Running experiment with depth={depth}, dropout={dropout}"
            )
            objective(params)


if __name__ == "__main__":
    main()
