## Hypothesis

I expected that different hyperparameters influence each other.
More epochs, more units, and deeper models should first improve performance, but after some point cause overfitting.
I also expected that regularisation, such as dropout or larger batch sizes, would reduce overfitting.
Finally, I expected adaptive optimizers like Adam or RMSprop to perform better than SGD.

## Experimental Setup

I changed one hyperparameter at a time and kept the others fixed:

- Epochs: 5 and 10
- Hidden units: 16 to 1024 (powers of two)
- Batch size: 4 to 128
- Model depth: 2 vs 3 linear layers
- Learning rate: 1e-2 to 1e-5
- Optimizer: SGD, Adam, RMSprop
Training and test loss were logged using TensorBoard 

## Results

### Epochs
Using 10 epochs gives lower training loss than 5 epochs. However, test loss increases, which means the model starts to overfit. More epochs do not automatically mean a better model.

### Number of Units
Models with less than 128 units clearly underfit. The lowest test loss is found between 256 and 512 units. Using more units than this increases training time but does not improve performance.

### Batch Size
Very small batch sizes cause unstable training and higher test loss. Batch sizes of 64 and 128 perform best. They give stable training and good generalisation.

### Model Depth
Adding a third linear layer makes the model worse. Test loss increases and training takes longer. The original two-layer model already has enough capacity.

### Learning Rate
Very small learning rates converge too slowly and underfit. Very large learning rates converge fast but to worse solutions. The best performance is achieved with a learning rate of 1e-4.

### Optimizers
SGD performs poorly within the limited number of epochs. Adam and RMSprop converge much faster and achieve much lower test loss. RMSprop performs slightly better than Adam, but the difference is small.

### Dropout and Depth
The hypothesis was that dropout would reduce overfitting in deeper models. This is not confirmed by the results. Dropout reduces performance for all tested model depths and causes underfitting.

### Reflection
These experiments show that increasing model complexity does not always improve results. There is a clear optimal range for model size and training settings. Optimizer choice has more impact than adding layers or units. Regularisation methods such as dropout are not always useful and depend on the dataset.

## Conclusion

The best performance is achieved with:

- Moderate number of units (256–512)
- Batch size of 64 or 128
- Learning rate around 1e-4
- Adaptive optimizers (Adam or RMSprop)

More epochs, deeper models, or dropout do not improve generalisation in this case.
This shows the importance of controlled experiments and critical evaluation instead of blindly increasing model complexity.