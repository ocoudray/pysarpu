"""Main module."""
import numpy as np
import pickle
from tqdm.auto import tqdm

class PUClassifier:
    '''
    PU learning classification model under unknown propensity.
    This model works by specifying a model on the classification and on the propensity and estimates parameters using EM algorithm (SAR-EM, Bekker et al.)

    :param cmodel: an instance of class :class:`pysarpu.classification.Classifier` representing the classification model. This package includes two types of classification models: logistic regression (accessible through :class:`pysarpu.classification.LinearLogisticRegression`) and linear discriminant analysis (accessible through :class:`pysarpu.classification.LinearDiscriminantClassifier`)
    :type cmodel: :class:`pysarpu.classification.Classifier`
    :param emodel: an instance of class :class:`pysarpu.propensity.Propensity` representing the propensity model. This package includes multiple pre-implemented propensity models: logistic propensity (:class:`pysarpu.propensity.LogisticPropensity`), log-normal propensity (:class:`pysarpu.propensity.LogProbitPropensity`) and Gumbel propensity (:class:`pysarpu.propensity.GumbelPropensity`)
    :type emodel: :class:`pysarpu.propensity.Propensity`
    :param da: whether the classification model is a discriminant analysis type model (`True`) or not (`False`). Indeed, the likelihood maximized is not the same in these two settings. Default: `False`.
    :type da: :class:`bool`, optional

    :return: Return an instance of PU learning model (not yet initialized).
    :rtype: :class:`pysarpu.PUClassifier`
    '''
    def __init__(self, cmodel, emodel, da=False):
        '''
        Constructor for :class:`pysarpu.PUClassifier`
        '''
        self.cmodel = cmodel
        self.emodel = emodel
        self.da = da
        self.history = []
    
    def initialization(self, Xc, Xe, Y, w=1.):
        '''
        Initialization of parameters for both classification and propensity models before running EM algorithm. The parameters of each models are initialized following their respective method: see `initialization` methods for `cmodel` and `emodel`.
        
        :param Xc: covariate matrix for classification. The parameters of `cmodel` will be initialized in agreement with the dimension of the entry data $d_1$.
        :type Xc: `numpy.array` of shape $(n,d_1)$
        :param Xe: covariate matrix for propensity. The parameters of `emodel` will be initialized in agreement with the dimension of the entry data $d_2$.
        :type Xe: `numpy.array` of shape $(n,d_2)$
        :param Y: observed labels. Only used in the computation of the initial log-likelihood.
        :type Y: `numpy.array` vector of size $n$.

        :return: `None`
        '''
        self.cmodel.initialization(Xc, w)
        self.emodel.initialization(Xe, w)
        self.history = [self.loglikelihood(Xc, Xe, Y)]

    
    def __str__(self):
        s = 'PU classifier \n'
        s += self.cmodel.__str__()
        s += '\n'
        s += self.emodel.__str__()
        return s

    def __repr__(self):
        return self.__str__()
    
    def eta(self, Xc):
        return self.cmodel.eta(Xc)

    def logeta(self, Xc):
        return self.cmodel.logeta(Xc)

    def e(self, Xe):
        '''
        Propensity function using the current parameters of propensity model `emodel`.

        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` of shape $(n,d_2)$

        :return: vector of propensity scores.
        :rtype: `numpy.array` of size $n$
        '''
        return self.emodel.e(Xe)

    def loge(self, Xe):
        '''
        Logarithm of propensity function using the current parameters of propensity model `emodel`.

        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` of shape $(n,d_2)$

        :return: vector of log-propensity scores.
        :rtype: `numpy.array` of size $n$
        '''
        return self.emodel.loge(Xe)
    
    def logp(self, Xe):
        # try:
        return self.emodel.logp(Xe)
        # except:
        #     print('Unavailable for this propensity model')

    def log1e(self, Xe):
        return self.emodel.log1e(Xe)
    
    def log1etae(self, Xc, Xe):
        leta, le = self.logeta(Xc), self.loge(Xe)
        approx = (leta + le <=-1e-10)
        return approx*np.nan_to_num(np.log1p(-np.exp(leta + le))) + (1-approx)*np.nan_to_num(np.log(-leta-le))
    
    def predict_cproba(self, Xc):
        '''
        Class probability predictions using the parameters of the classification model.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.

        :return: posterior class probabilities.
        :rtype: `numpy.array` vector of size $n$
        '''
        return self.cmodel.eta(Xc)

    def predict_clogproba(self, Xc):
        '''
        Class log-probability predictions using the parameters of the classification model.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.

        :return: posterior class log-probabilities.
        :rtype: `numpy.array` vector of size $n$
        '''
        return self.cmodel.logeta(Xc)
    
    def predict_c(self, Xc, threshold=0.5):
        '''
        Class binary predictions using the parameters of the classification model.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param threshold: decision threshold defining the decision rule.
        :type threshold: `float`, optional (in $[0,1]$)

        :return: class predictions.
        :rtype: `numpy.array` binary vector of size $n$
        '''
        return (self.predict_cproba(Xc)>=threshold).astype(int)

    def predict_eproba(self, Xe):
        return self.emodel.e(Xe)

    def predict_elogproba(self, Xe):
        return self.emodel.loge(Xe)
    
    def predict_e(self, Xe, threshold=0.5):
        return (self.predict_eproba(Xe)>=threshold).astype(int)
    
    def predict_proba(self, Xc, Xe):
        '''
        Label probability predictions based on the classification model `cmodel` and the propensity model `emodel`. Note that this is different from method `predict_cproba` which returns class probabilities instead.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.

        :return: posterior label probabilities.
        :rtype: `numpy.array` vector of size $n$
        '''
        return self.predict_cproba(Xc)*self.predict_eproba(Xe)

    def predict_logproba(self, Xc, Xe):
        '''
        Label log-probability predictions based on the classification model `cmodel` and the propensity model `emodel`. Note that this is different from method `predict_clogproba` which returns class log-probabilities instead.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.

        :return: posterior label log-probabilities.
        :rtype: `numpy.array` vector of size $n$
        '''
        return self.predict_clogproba(Xc) + self.predict_elogproba(Xe)

    def predict(self, Xc, Xe, threshold=0.5):
        '''
        Label binary predictions based on the classification model `cmodel` and the propensity model `emodel`. Note that this is different from method `predict_c` which returns class predictions instead.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.
        :param threshold: decision threshold defining the decision rule.
        :type threshold: `float`, optional (in $[0,1]$)

        :return: label binary predictions.
        :rtype: `numpy.array` binary vector of size $n$
        '''
        return (self.predict_proba(Xc, Xe)>=threshold).astype(int)
    
    def loglikelihood(self, Xc, Xe, Y, w=1.):
        '''
        Log-likelihood function given the current parameters of classification and propensity models. Note that the funciton returns the mean of individual dlog-likelihoods (instead of the usual sum).

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.
        :param Y: observed labels. Only used in the computation of the initial log-likelihood.
        :type Y: `numpy.array` vector of size $n$.
        :params w: individual weights (experimental, not tested). Apply weights to observations in the computation of the likelihood.
        :type w: either `float` (`1.`, default) or `numpy.array` of size $n$, optional.

        :return: log-likelihood.
        :rtype: `float`      
        '''
        if self.da:
            return self.loglikelihood_da(Xc, Xe, Y, w)
        else:
            return self.loglikelihood_lr(Xc, Xe, Y, w)
    
    def loglikelihood_lr(self, Xc, Xe, Y, w=1.):
        # if self.pdf == False:
        #     return np.mean(w*(Y*self.logeta(Xc) + Y*self.loge(Xe) + (1-Y)*self.log1etae(Xc, Xe)))
        # else:
        #     return np.mean(w*(Y*self.logeta(Xc) + Y*self.logp(Xe) + (1-Y)*self.log1etae(Xc, Xe)))
        return np.mean(w*(Y*self.logeta(Xc) + Y*self.loge(Xe) + (1-Y)*self.log1etae(Xc, Xe)))

    def loglikelihood_da(self, Xc, Xe, Y, w=1.):
        return np.mean(Y*(np.log(self.cmodel.pdf_pos(Xc)) + self.emodel.loge(Xe)) + (1-Y)*np.log(self.cmodel.pdf_pos(Xc)*(1-self.emodel.e(Xe)) + self.cmodel.pdf_neg(Xc)))
    
    def expectation(self, Xc, Xe, Y):
        '''
        Compute the expectation step of EM algorithm, return the probabilities for every instance to be of positive class given the observed labels.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.
        :param Y: observed labels. Only used in the computation of the initial log-likelihood.
        :type Y: `numpy.array` vector of size $n$.

        :return: posterior probabilities
        :rtype: `np.array` vector of size $n$
        '''
        l1 = self.logeta(Xc) + self.log1e(Xe)
        l2 = self.log1etae(Xc, Xe)
        p = np.min(np.concatenate([np.exp(l1-l2)[:,np.newaxis], np.ones((len(Y),1))], axis=1), axis=1)
        return Y + (1-Y)*p

    def maximisation(self, Xc, Xe, Y, gamma, w=1., warm_start=True, balance=False):
        '''
        Compute the maximisation step of EM algorithm, update the model parameters in both classification and propensity models.

        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.
        :param Y: observed labels. Only used in the computation of the initial log-likelihood.
        :type Y: `numpy.array` vector of size $n$.
        :gamma: posterior probabilities obtained in the expectation step.
        :type gamma: `numpy.array` of size $n$
        :params w: individual weights (experimental, not tested). Apply weights to observations in the computation of the likelihood.
        :type w: either `float` (`1.`, default) or `numpy.array` of size $n$, optional.

        :return: `None`
        '''
        self.prev_params_c, self.prev_params_e = self.cmodel.params.copy(), self.emodel.params.copy()
        self.cmodel.fit(Xc, gamma, w, warm_start)
        self.emodel.fit(Xe, gamma, Y, w, warm_start, balance)
    
    def _fit(self, Xc, Xe, Y, w=1., tol=1e-6, max_iter=1e4, warm_start=False, balance=False):
        if not warm_start:
            self.initialization(Xc, Xe, Y, Y, w)
        it = 0
        delta = 1
        last_ll = self.loglikelihood(Xc, Xe, Y, w)
        warm_start = False
        while abs(delta) > tol and it < max_iter:
            if delta<0:
                # print("Attention")
                self.cmodel.params = self.prev_params_c.copy()
                self.emodel.params = self.prev_params_e.copy()
            gamma = self.expectation(Xc, Xe, Y)
            self.maximisation(Xc, Xe, Y, gamma, w, warm_start, balance)
            delta = self.loglikelihood(Xc, Xe, Y, w) - last_ll
            last_ll = self.loglikelihood(Xc, Xe, Y, w)
            self.history.append(last_ll)
            # print(last_ll)
            it += 1
            warm_start = True

    def fit(self, Xc, Xe, Y, w=1., tol=1e-6, max_iter=1e4, warm_start=False, balance=False, n_init=20, iter_init=20):
        '''
        Estimation of PU learning model parameters (classifier and propensity) through EM algorithm. Multiple random initialization are considered and trained over a few iterations. Then, only the one achieving the best log-likelihood is considered and trained until convergence.
        
        :param Xc: covariate matrix for classification.
        :type Xc: `numpy.array` with shape $(n, d_1)$.
        :param Xe: covariate matrix for propensity.
        :type Xe: `numpy.array` with shape $(n, d_2)$.
        :param Y: observed labels. Only used in the computation of the initial log-likelihood.
        :type Y: `numpy.array` vector of size $n$.
        :params w: individual weights (experimental, not tested). Apply weights to observations in the computation of the likelihood.
        :type w: either `float` (`1.`, default) or `numpy.array` of size $n$, optional.
        :param tol: tolerance parameter. Once the increase in the log-likelihood is below `tol`, the algorithm stops (default `1e-6`).
        :type tol: float, optional
        :param max_iter: maximum number of iterations (default: `1e4`)
        :type max_iter: int, optional
        :param warm_start: indicates whether current parameters can be used for initialization (`True`) or if they should be re-initialized before estimation (default `False`).
        :type warm_start: bool, optional
        :param balance: re-balance weights when fitting the propensity model in the maximization (experimental, potentially interesting in highly unbalanced situations). Default: `False`.
        :type balance: bool, optional
        :param n_init: number of initialization to consider in the Small EM initialization strategy (default: `n_init=20`)
        :type n_init: int, optional
        :param iter_init: maximum number of iterations to consider for each initialization (default: `20`).
        :type iter_init: int, optional

        :return: `None`
        '''
        if not warm_start:
            self.initialization(Xc, Xe, Y, Y, w)
        optimal_ll = self.loglikelihood(Xc, Xe, Y)
        optimal_params_c = self.cmodel.params.copy()
        optimal_params_e = self.emodel.params.copy()
        for k in tqdm(range(n_init)):
            self._fit(Xc, Xe, Y, w=w, tol=tol, max_iter=iter_init, warm_start=False, balance=balance)
            ll = self.loglikelihood(Xc, Xe, Y)
            if ll > optimal_ll:
                optimal_ll = ll
                optimal_params_c = self.cmodel.params.copy()
                optimal_params_e = self.emodel.params.copy()
        self.cmodel.params = optimal_params_c.copy()
        self.emodel.params = optimal_params_e.copy()
        self._fit(Xc, Xe, Y, w=w, tol=tol, max_iter=max_iter, warm_start=True, balance=balance)

    
    def save(self, path):
        '''
        Saving PU learning model with current parameters as a binary file (rely on `pickle` library).
        
        :param path: path at which the model should be saved.
        :type path: `str`

        :return: `None`
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)