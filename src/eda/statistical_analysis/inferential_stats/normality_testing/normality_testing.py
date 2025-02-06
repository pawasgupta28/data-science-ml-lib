import numpy as np
import pandas as pd
import scipy.stats as stats

class NormalityTesting:
    def __init__(self, data):
        """Initialize with a Pandas DataFrame and a list of columns for normality testing."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data should be a Pandas DataFrame")
        
        self.data = data
    
    def shapiro_wilk_test(self):
        """Performs the Shapiro-Wilk test for normality."""
        return {col: stats.shapiro(self.data[col]) for col in self.data.columns}
    
    def anderson_darling_test(self):
        """Performs the Anderson-Darling test for normality."""
        return {col: stats.anderson(self.data[col], dist='norm') for col in self.data.columns}
    
    def kolmogorov_smirnov_test(self):
        """Performs the Kolmogorov-Smirnov test against a normal distribution."""
        return {col: stats.kstest(self.data[col], 'norm') for col in self.data.columns}
    
    def dagostino_k2_test(self):
        """Performs the D'Agostino and Pearson's K-squared test for normality."""
        return {col: stats.normaltest(self.data[col]) for col in self.data.columns}
    
    def summary(self):
        """Returns a dictionary containing all normality test results."""
        return {
            "Shapiro-Wilk Test": self.shapiro_wilk_test(),
            "Anderson-Darling Test": self.anderson_darling_test(),
            "Kolmogorov-Smirnov Test": self.kolmogorov_smirnov_test(),
            "D'Agostino K2 Test": self.dagostino_k2_test()
        }

# Example Usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "A": np.random.normal(0, 1, 100),
        "B": np.random.exponential(1, 100),
        "C": np.random.uniform(-1, 1, 100)
    })
    
    norm_test = NormalityTesting(df, ["A", "B", "C"])
    print(norm_test.summary())
