## 1. Introduction
Deep learning models involve many design choices, of which architectural decisions are among the most influential. While automated hyperparameter tuning methods can be useful, they may obscure the relationship between model capacity and performance if used without reflection. The goal of this assignment is therefore not to optimise performance, but to demonstrate a scientific approach to model design: formulating hypotheses based on theory, designing controlled experiments, analysing results, and iterating accordingly.
In this study, I investigate how model capacity affects performance and generalization in a convolutional neural network (CNN) trained on the Flowers-102 dataset. The focus is placed on architectural hyperparameters, specifically the size of dense layers, while keeping optimization-related parameters fixed to maintain interpretability.

## 2. Theory and Hypothesis
From learning theory, the bias–variance trade-off states that models with insufficient capacity underfit the data, while overly complex models may overfit and generalize poorly. In neural networks, model capacity is largely determined by architectural choices such as the number of hidden units in dense layers.
Based on this theory, the following hypothesis is formulated:
Hypothesis
Increasing the number of hidden units in the dense layers improves validation performance up to a certain point, after which performance gains diminish or plateau.

## 3. Experimental Setup
Dataset and Model
The Flowers-102 dataset is used, consisting of RGB images resized to 64×64 pixels across 102 classes. A custom CNN is implemented with a fixed convolutional feature extractor followed by two dense layers and a classification layer. The convolutional part is kept constant to isolate the effect of dense-layer capacity.
Training Configuration
To ensure fair comparison, the following parameters are fixed across all experiments:
•	Optimizer: Adam
•	Learning rate: 1e-3
•	Batch size: 32
•	Epochs: 5
•	Training steps per epoch: 50
•	Validation steps per epoch: 25
The limited training budget is intentional: the objective is to compare relative performance, not to fully optimise each model.
MLflow is used to track experiments. Each configuration is logged as a separate run with hyperparameters stored as tags and metrics logged per epoch.

## 4. Experimental Design
An initial exploratory phase was used to identify stable training behaviour. Learning rate experiments showed that 1e-3 consistently outperformed smaller values under the given constraints. Since learning rate primarily affects optimization rather than representational capacity, it was fixed in the main experiment.
The main experiment systematically varies the number of hidden units in the dense layers:
•	units1: {64, 256, 512}
•	units2: {64, 256, 512}
The number of convolutional filters is fixed at 32. This results in nine configurations. For configurations with multiple runs, results are aggregated using the mean validation performance.

## 5. Results and Analysis
Validation loss and accuracy are visualized using heatmaps to analyse interactions between units1 and units2. 
The results show a clear and structured pattern. Models with units1 = 64 consistently perform poorly, indicating underfitting. Increasing units1 to 256 leads to a substantial improvement in both validation loss and accuracy. Further increasing units1 to 512 yields smaller gains, suggesting diminishing returns.
The second dense layer (units2) has a weaker effect on performance. While increasing units2 can improve results, its impact is secondary compared to units1. This suggests that representational capacity earlier in the classifier is more important than additional capacity in later layers.
Accuracy and loss trends are consistent, and no extreme overfitting is observed. Training and validation losses remain relatively close, indicating that models are still partially underfitting the dataset, which is expected given the dataset complexity and limited training budget.

## 6. Conclusion
The results support the hypothesis that increasing model capacity improves performance up to a point, after which gains diminish. The size of the first dense layer has the strongest influence on validation performance, while the second dense layer plays a more limited role.
These findings align with theoretical expectations from the bias–variance trade-off. The experiment demonstrates how controlled architectural changes can be systematically analysed using a scientific approach. While absolute performance remains modest, the observed trends are clear and robust, providing meaningful insight into the relationship between model capacity and generalization.

