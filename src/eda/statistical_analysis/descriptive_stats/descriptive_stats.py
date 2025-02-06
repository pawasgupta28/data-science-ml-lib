import numpy as np
import pandas as pd
import scipy.stats as stats

class DescriptiveStatistics:
    def __init__(self, data):
        if not isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray)):
            raise ValueError("Data should be a Pandas DataFrame, Series, list, or NumPy array")
        self.data = pd.DataFrame(data)
    
    def mean(self):
        """Returns the mean of the dataset."""
        return self.data.mean()
    
    def median(self):
        """Returns the median of the dataset."""
        return self.data.median()
    
    def mode(self):
        """Returns the mode of the dataset."""
        return self.data.mode().iloc[0] if not self.data.mode().empty else None
    
    def variance(self):
        """Returns the variance of the dataset."""
        return self.data.var()
    
    def std_dev(self):
        """Returns the standard deviation of the dataset."""
        return self.data.std()
    
    def iqr(self):
        """Returns the Interquartile Range (IQR) of the dataset."""
        return f'{self.data.quantile(0.25)} - {self.data.quantile(0.75)}'
    
    def skewness(self):
        """Returns the skewness of the dataset."""
        return self.data.skew()
    
    def kurtosis(self):
        """Returns the kurtosis of the dataset."""
        return self.data.kurt()
    
    def summary(self):
        """Returns a DataFrame containing all descriptive statistics."""
        return pd.DataFrame({
            "Mean": self.mean(),
            "Median": self.median(),
            "Mode": self.mode(),
            "Variance": self.variance(),
            "Standard Deviation": self.std_dev(),
            "IQR": self.iqr(),
            "Skewness": self.skewness(),
            "Kurtosis": self.kurtosis()
        })

# Example Usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "B": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    })
    stats_obj = DescriptiveStatistics(df)
    print(stats_obj.summary())