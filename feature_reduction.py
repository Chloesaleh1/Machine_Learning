import pandas as pd
import statsmodels.api as sm

class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05): # 9 pts
        '''
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        '''
        first = data.columns.tolist()
        best = []
        i = len(first)
        while (i>0):
            remaining_features = list(set(first)-set(best))
            new_pval = pd.Series(index=remaining_features)
            for n in remaining_features:
                model = sm.OLS(target, sm.add_constant(data[best+[n]])).fit()
                new_pval[n] = model.pvalues[n]
            min_pval = new_pval.min()
            if(min_pval<significance_level):
                best.append(new_pval.idxmin())
            else:
                break
        return best

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): # 9 pts
        '''
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        '''
        raise NotImplementedError
