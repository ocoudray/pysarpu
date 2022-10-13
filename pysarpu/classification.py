import statsmodels.api as sm
import numpy as np
import scipy.stats as scs

class Classifier:
    '''
    Instance of classifier to be used in a PU classification model
    '''
    def __init__(self):
        self.params = None

    def initialization(self, Xc):
        self.params = np.random.rand(Xc.shape[1] + 1)

    def eta(self, Xc):
        return 0.5*np.ones(Xc.shape[0])

    def logeta(self, Xc):
        return np.zeros(Xc.shape[0])

    def predict_proba(self, Xc):
        return self.eta(Xc)

    def predict_logproba(self, Xc):
        return self.logeta(Xc)

    def predict(self, Xc, threshold=0.5):
        return (self.predict_proba(Xc)>=threshold).astype(int)

class LinearLogisticRegression(Classifier):
    def __init__(self):
        super(LinearLogisticRegression, self).__init__()

    def __str__(self):
        if self.params is not None:
            s = 'theta = {}'.format(self.params)
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()

    def initialization(self, Xc, gamma, w=1.):
        self.params = np.random.randn(Xc.shape[1]+1)

    def fit(self, Xc, gamma, w=1., warm_start=True):
        # add intercept
        # Xc = np.concatenate([np.ones((Xc.shape[0],1)), Xc], axis=1)
        Xc = np.concatenate([np.ones(Xc.shape[:-1]+(1,)), Xc], axis=-1)
        if self.params is None:
            self.initialization(Xc, gamma, w)
        if warm_start == False:
            start_params = None
        else:
            start_params = self.params.copy()
        Xc_ = np.concatenate([Xc,Xc], axis=0)
        Z_ = np.concatenate([np.ones(len(gamma)),np.zeros(len(gamma))], axis=0)
        q1, q2 = sum(gamma)/len(gamma), 1-sum(gamma)/len(gamma)
        c1, c2 = 1/q1/(1/q1 + 1/q2), 1/q2/(1/q1 + 1/q2)
        w_ = np.concatenate([gamma, (1-gamma)], axis=0)
        self.model = sm.GLM(Z_, Xc_, freq_weights=w_, family=sm.families.Binomial())
        # self.model = LogisticRegression(penalty='none', max_iter=1e5)
        # else:
        #     self.model.exog = np.concatenate([Xc,Xc], axis=0)
        #     self.model.endog = np.concatenate([np.ones(len(gamma)),np.zeros(len(gamma))], axis=0)
        #     self.model.freq_weights = np.concatenate([w*gamma, w*(1-gamma)], axis=0)
        #     # params_init = self.params.copy()
        #     params_init = None
        self.res = self.model.fit(start_params=start_params, disp=0)
        # # else:
        # #     self.res = self.model.fit_regularized(alpha=self.C, start_params=start_params, maxiter=int(1e3))
        self.params = self.res.params.copy()
        # self.model.fit(Xc_, Z_, sample_weight=w_)
        # self.params = self.model.coef_[0]

    def eta(self, Xc):
        # add intercept
        # Xc = np.concatenate([np.ones((Xc.shape[0],1)), Xc], axis=1)
        Xc = np.concatenate([np.ones(Xc.shape[:-1]+(1,)), Xc], axis=-1)
        if self.params is None:
            print('Please initialize parameters first')
            return
        return scs.logistic.cdf(np.dot(Xc, self.params))

    def logeta(self, Xc):
        # add intercept
        # Xc = np.concatenate([np.ones((Xc.shape[0],1)), Xc], axis=1)
        Xc = np.concatenate([np.ones(Xc.shape[:-1]+(1,)), Xc], axis=-1)
        if self.params is None:
            print('Please initialize parameters first')
            return
        return scs.logistic.logcdf(np.dot(Xc, self.params))


class LinearDiscriminantClassifier(Classifier):
    def __init__(self):
        super(LinearDiscriminantClassifier).__init__()
    
    def initialization(self, Xc, gamma, w=1.):
        self.params = {'pi':np.random.rand(), 'mu_0':np.random.randn(Xc.shape[1]), 'mu_1':np.random.randn(Xc.shape[1]), 'Sigma':1/len(Xc)*np.dot((Xc-Xc.mean(axis=0)).T, Xc-Xc.mean())}
        # self.params = {'pi':np.mean(gamma), 'mu_0':np.mean(Xc[gamma==0,:], axis=0), 'mu_1':np.mean(Xc[gamma==1,:], axis=0), 'Sigma':1/len(Xc)*np.dot((Xc-Xc.mean(axis=0)).T, Xc-Xc.mean())}

        # self.Sigma = 1/len(Xc)*np.dot((Xc-Xc.mean(axis=0)).T, Xc-Xc.mean())
        # self.Sigma = np.eye(Xc.shape[1])
        # self.params = {'pi':gamma.mean(), 'mu_0':Xc[gamma==0,:].mean(axis=0), 'mu_1':Xc[gamma==1,:].mean(axis=0)}
        # X0, X1 = np.dot((Xc-self.params['mu_0']).T,(1-gamma[:,np.newaxis])*(Xc-self.params['mu_0'])), np.dot((Xc-self.params['mu_1']).T,gamma[:,np.newaxis]*(Xc-self.params['mu_1']))
        # self.params['Sigma'] = 1/len(Xc)*(X0+X1)



    def __str__(self):
        if self.params is not None:
            s = r'$\mu_0 = {}$'.format(self.params['mu_0'])
            s += r'$\mu_1 = {}$'.format(self.params['mu_1'])
            s += r'$\pi = {}$'.format(self.params['pi'])
            s += r'$\Sigma = {}$'.format(self.params['Sigma'])
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()
    
    def fit(self, Xc, gamma, w=1., warm_start=True):
        self.params['pi'] = np.mean(gamma)
        self.params['mu_1'] = np.mean(gamma[:,np.newaxis]*w*Xc, axis=0)/np.mean(gamma)
        self.params['mu_0'] = np.mean((1-gamma)[:,np.newaxis]*w*Xc, axis=0)/np.mean(1-gamma)
        X0, X1 = np.dot((Xc-self.params['mu_0']).T,(1-gamma[:,np.newaxis])*(Xc-self.params['mu_0'])), np.dot((Xc-self.params['mu_1']).T,gamma[:,np.newaxis]*(Xc-self.params['mu_1']))
        self.params['Sigma'] = 1/len(Xc)*(X0+X1)
    
    def eta(self, Xc):
        f1 = scs.multivariate_normal.pdf(Xc, mean=self.params['mu_1'], cov=self.params['Sigma'])
        f0 = scs.multivariate_normal.pdf(Xc, mean=self.params['mu_0'], cov=self.params['Sigma'])
        return self.params['pi']*f1/(self.params['pi']*f1 + (1-self.params['pi'])*f0)
    
    def logeta(self, Xc):
        p0 = (np.dot((Xc-self.params['mu_0']), np.linalg.inv(self.params['Sigma']))*(Xc-self.params['mu_0'])).sum(axis=1)
        p1 = (np.dot((Xc-self.params['mu_1']), np.linalg.inv(self.params['Sigma']))*(Xc-self.params['mu_1'])).sum(axis=1)
        x = -np.log((1-self.params['pi'])/self.params['pi']) + 0.5 * p0 - 0.5 * p1
        return scs.logistic.logcdf(x)
    
    def pdf_pos(self, Xc):
        return self.params['pi']*scs.multivariate_normal.pdf(Xc, mean=self.params['mu_1'], cov=self.params['Sigma'])

    def pdf_neg(self, Xc):
        return (1-self.params['pi'])*scs.multivariate_normal.pdf(Xc, mean=self.params['mu_0'], cov=self.params['Sigma'])