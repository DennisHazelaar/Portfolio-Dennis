---
layout: default
---

### Hypothesis
As model depth increases, overfitting becomes more pronounced; however, the application of dropout significantly mitigates this effect. In shallow networks, dropout negatively impacts validation performance due to underfitting, whereas in deeper networks it improves generalization by reducing variance.
Hyperparameter	Value
Model depth	2,4,6
Dropout rate	0.0, 0.3, 0.5


### Results
The experimental results contradict the initial hypothesis. While dropout was expected to mitigate overfitting in deeper networks, no evidence of improved generalization was observed. Instead, dropout consistently degraded performance across all tested depths.
 
