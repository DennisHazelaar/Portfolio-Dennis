---
layout: default
---

## Introduction

Recurrent Neural Networks (RNNs) are widely used for sequential data because they can model temporal dependencies. However, standard RNN architectures often have difficulty learning local temporal patterns and may require large model sizes to perform well. The goal of this study was to improve the performance of a baseline RNN model and achieve a classification accuracy above 90%. Different model architectures were evaluated, including GRU and LSTM networks, as well as models that combine Conv1D layers with recurrent layers.

## Baseline Recurrent Models

As a starting point, several GRU and LSTM models were trained with different hidden sizes (64–1024 units), numbers of layers (1–3), and dropout values. Increasing the hidden size generally reduced the training loss, but this did not consistently improve test performance. In many cases, larger models showed signs of overfitting, with low training loss but higher test loss.

LSTM models required more training time and did not outperform GRU models. These results indicate that the dataset does not require the additional complexity of LSTM units and that GRUs are sufficient to model the temporal structure of the data.

## Convolutional Feature Extraction

To improve the model’s ability to learn local temporal patterns, Conv1D layers were added before the recurrent layers. The convolutional layers operate along the time dimension and learn short-range patterns that are passed to the GRU.

Models that combined Conv1D layers with GRU units consistently achieved lower test loss than models without convolutional layers. The best performance was obtained using a Conv1D layer with 32 output channels followed by a GRU with 256 hidden units. This shows that learning local features before applying recurrent processing improves generalization and training stability.

## Effect of Kernel Size

Different convolution kernel sizes (3, 5, and 7) were tested. Smaller kernels produced better results, while larger kernels led to worse test performance. This suggests that the important patterns in the dataset are short in time and that large kernels may smooth out useful information.

## Model Depth and Regularization

Increasing the number of recurrent layers beyond two did not lead to performance improvements and often increased training time. Dropout was also evaluated as a regularization method, but in many experiments it reduced test performance. Since Batch Normalization was already used in the convolutional layers, additional regularization through dropout appeared unnecessary.

## Final Model and Conclusion

The final model consisted of a Conv1D layer followed by a GRU with a moderate hidden size and a small number of layers. This architecture achieved a test loss corresponding to an accuracy well above 90%.

In conclusion, the experiments show that improving feature extraction using convolutional layers is more effective than increasing the size or depth of recurrent models. The combination of Conv1D and GRU layers provided the best balance between performance, simplicity, and training efficiency.