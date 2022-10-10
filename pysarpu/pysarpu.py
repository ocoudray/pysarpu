"""Main module."""
import numpy as np
import pickle

class PUClassifier:
    '''
    PU learning classification model under unknown propensity.
    This model works by specifying a model on the classification and on the propensity and estimates parameters using EM algorithm (SAR-EM, Bekker et al.)
    '''
    def __init__(self, cmodel, emodel, pdf=False, da=False):
        '''
        Initialization of PU classifier. Two parameters are required:
        - cmodel: an instance of classification model
        - emodel: an instance of propensity model
        '''
        self.cmodel = cmodel
        self.emodel = emodel
        self.pdf = pdf
        self.da = da
        self.history = []
    
    def initialization(self, Xc, Xe, Y, gamma, w=1.):
        '''
        Initialization of parameters for both model before running EM algorithm
        '''
        self.cmodel.initialization(Xc, Y, w)
        self.emodel.initialization(Xe, gamma, Y, w)
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
        return self.emodel.e(Xe)

    def loge(self, Xe):
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
        return self.cmodel.eta(Xc)

    def predict_clogproba(self, Xc):
        return self.cmodel.logeta(Xc)
    
    def predict_c(self, Xc, threshold=0.5):
        return (self.predict_cproba(Xc)>=threshold).astype(int)

    def predict_eproba(self, Xe):
        return self.emodel.e(Xe)

    def predict_elogproba(self, Xe):
        return self.emodel.loge(Xe)
    
    def predict_e(self, Xe, threshold=0.5):
        return (self.predict_eproba(Xe)>=threshold).astype(int)
    
    def predict_proba(self, Xc, Xe):
        return self.predict_cproba(Xc)*self.predict_eproba(Xe)

    def predict_logproba(self, Xc, Xe):
        return self.predict_clogproba(Xc) + self.predict_elogproba(Xe)

    def predict(self, Xc, Xe, threshold=0.5):
        return (self.predict_proba(Xc, Xe)>=threshold).astype(int)
    
    def loglikelihood(self, Xc, Xe, Y, w=1.):
        if self.da:
            return self.loglikelihood_da(Xc, Xe, Y, w)
        else:
            return self.loglikelihood_lr(Xc, Xe, Y, w)
    
    def loglikelihood_lr(self, Xc, Xe, Y, w=1.):
        '''
        Compute the loglikelihood of the model given observations:
        - Xc: the feature vector used in the classification model
        - Xe: the feature vector used in the propensity model
        - Y: observed noisy PU labels. Y=1 (labeled) or Y=0 (unlabeled)
        - w: classification weights (default=1. else, should be an array of positive float with the same size as Y)
        '''
        if self.pdf == False:
            return np.mean(w*(Y*self.logeta(Xc) + Y*self.loge(Xe) + (1-Y)*self.log1etae(Xc, Xe)))
        else:
            return np.mean(w*(Y*self.logeta(Xc) + Y*self.logp(Xe) + (1-Y)*self.log1etae(Xc, Xe)))

    def loglikelihood_da(self, Xc, Xe, Y, w=1.):
        '''
        Compute the loglikelihood of the model given observations:
        - Xc: the feature vector used in the classification model
        - Xe: the feature vector used in the propensity model
        - Y: observed noisy PU labels. Y=1 (labeled) or Y=0 (unlabeled)
        - w: classification weights (default=1. else, should be an array of positive float with the same size as Y)
        '''
        return np.mean(Y*(np.log(self.cmodel.pdf_pos(Xc)) + self.emodel.loge(Xe)) + (1-Y)*np.log(self.cmodel.pdf_pos(Xc)*(1-self.emodel.e(Xe)) + self.cmodel.pdf_neg(Xc)))
    
    def expectation(self, Xc, Xe, Y):
        '''
        Compute the expectation step of EM algorithm, return the probabilities for every instance to be of positive class:
        - Xc: the feature vector used in the classification model
        - Xe: the feature vector used in the propensity model
        - Y: observed noisy PU labels. Y=1 (labeled) or Y=0 (unlabeled)
        '''
        l1 = self.logeta(Xc) + self.log1e(Xe)
        l2 = self.log1etae(Xc, Xe)
        p = np.min(np.concatenate([np.exp(l1-l2)[:,np.newaxis], np.ones((len(Y),1))], axis=1), axis=1)
        return Y + (1-Y)*p

    def maximisation(self, Xc, Xe, Y, gamma, w=1., warm_start=True, balance=False):
        '''
        Compute the maximisation step of EM algorithm, update the model parameters in both classification and propensity models
        - Xc: the feature vector used in the classification model
        - Xe: the feature vector used in the propensity model
        - Y: observed noisy PU labels. Y=1 (labeled) or Y=0 (unlabeled)
        - gamma: probabilities obtained at expectation step
        '''
        self.prev_params_c, self.prev_params_e = self.cmodel.params.copy(), self.emodel.params.copy()
        self.cmodel.fit(Xc, gamma, w, warm_start)
        self.emodel.fit(Xe, gamma, Y, w, warm_start, balance)
    
    def fit(self, Xc, Xe, Y, w=1., tol=1e-6, max_iter=1e4, warm_start=False, balance=False):
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
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)