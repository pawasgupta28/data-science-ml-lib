import numpy as np
import pandas as pd
import scipy.stats as stats

class HypothesisTesting:
    def __init__(self, data1, data2=None):
        """Initialize with one or two datasets (Pandas Series, lists, or NumPy arrays)."""
        if not isinstance(data1, (pd.Series, list, np.ndarray)):
            raise ValueError("Data should be a Pandas Series, list, or NumPy array")
        self.data1 = np.array(data1)
        
        if data2 is not None:
            if not isinstance(data2, (pd.Series, list, np.ndarray)):
                raise ValueError("Data should be a Pandas Series, list, or NumPy array")
            self.data2 = np.array(data2)
        else:
            self.data2 = None
    
    def t_test(self, equal_var=True):
        """Performs an independent t-test (two-sample) or one-sample t-test if data2 is None."""
        if self.data2 is not None:
            return stats.ttest_ind(self.data1, self.data2, equal_var=equal_var)
        return stats.ttest_1samp(self.data1, 0)
    
    def paired_t_test(self):
        """Performs a paired t-test (dependent samples)."""
        if self.data2 is None:
            raise ValueError("Paired t-test requires two datasets")
        return stats.ttest_rel(self.data1, self.data2)
    
    def anova(self, *groups):
        """Performs a one-way ANOVA test for multiple groups."""
        return stats.f_oneway(self.data1, *groups)
    
    def chi_square(self, observed):
        """Performs a chi-square test on a contingency table."""
        return stats.chi2_contingency(observed)
    
    def ks_test(self, distribution='norm'):
        """Performs the Kolmogorov-Smirnov test for goodness of fit."""
        return stats.kstest(self.data1, distribution)
    
    def summary(self):
        """Returns a dictionary of test results."""
        results = {"One-Sample T-Test": self.t_test().pvalue}
        if self.data2 is not None:
            results["Two-Sample T-Test"] = self.t_test().pvalue
            results["Paired T-Test"] = self.paired_t_test().pvalue
        return results

# Example Usage
if __name__ == "__main__":
    data1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data2 = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    
    test_obj = HypothesisTesting(data1, data2)
    print(test_obj.summary())
