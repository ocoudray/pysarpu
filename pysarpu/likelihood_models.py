from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import scipy.stats as scs

class LogProbitSig(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights):
        self.weights = weights
        super(LogProbitSig, self).__init__(endog, exog)

    def loge(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logcdf(np.log(pred), scale=np.exp(params[-1]))

    def log1e(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logsf(np.log(pred), scale=np.exp(params[-1]))
    
    def loglike(self, params):
        endog = self.endog
        return (self.weights*endog*self.loge(params) + (1-endog)*self.weights*self.log1e(params)).sum()
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('sig')
        if start_params is None:
            # Reasonable starting values
            # start_params = np.zeros((self.exog.shape[1] + 1))
            start_params = -0.5 + np.random.rand(self.exog.shape[1] + 1)
        return super(LogProbitSig, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
class Gumbel(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights):
        self.weights = weights
        super(Gumbel, self).__init__(endog, exog)

    def loge(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.weibull_min.logcdf(pred, np.exp(params[-1]))

    def log1e(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.weibull_min.logsf(pred, np.exp(params[-1]))
    
    def loglike(self, params):
        endog = self.endog
        return (self.weights*endog*self.loge(params) + (1-endog)*self.weights*self.log1e(params)).sum()
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('b')
        if start_params is None:
            # Reasonable starting values
            # start_params = np.zeros((self.exog.shape[1] + 1))
            start_params = -0.5 + np.random.rand(self.exog.shape[1] + 1)
        return super(Gumbel, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)