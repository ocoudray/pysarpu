=======
Modules
=======

PU learning model
-----------------
.. autoclass:: pysarpu.PUClassifier
    :members:

Classification models
---------------------
Two classification models can be found in submodule `pysarpu.classification`:

- a Linear Logistic Regression model
- a Linear Discriminant Analysis model

These two models inherit from the general class `sklearn.classification.Classifier`.

.. autoclass:: pysarpu.classification.LinearLogisticRegression
    :members:
.. autoclass:: pysarpu.classification.LinearDiscriminantClassifier
    :members:

Propensity models
-----------------
Three propensity models are provided in submodule `pysarpu.propensity`:

- a Logistic Regression model
- a logistic function with log-normal link function
- a logistic function with Weibull link function

.. autoclass:: pysarpu.propensity.LogisticPropensity
    :members:
.. autoclass:: pysarpu.propensity.LogProbitPropensity
    :members:
.. autoclass:: pysarpu.propensity.GumbelPropensity
    :members: