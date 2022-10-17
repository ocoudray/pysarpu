=====
Usage
=====

Import
------

To use pysarpu in a project::

    import pysarpu

PU learning classification model can be imported as follows::

    from pysarpu import PU

The definition of a PU model requires the specification of a classification model and of a propensity model. Implementations can be found in sub-modules `pysarpu.classification` and `pysarpu.propensity`::
    
    from pysarpu.classification import LinearLogisticRegression, LinearDiscriminantClassifier
    from pysarpu.propensity import LogisticPropensity, LogProbitPropensity, GumbelPropensity


