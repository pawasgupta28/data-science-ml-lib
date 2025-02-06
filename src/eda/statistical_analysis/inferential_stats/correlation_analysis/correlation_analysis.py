import numpy as np
import pandas as pd
import scipy.stats as stats

class CorrelationTesting:
    def __init__(self, data):
        """Initialize with a Pandas DataFrame and a list of columns for correlation analysis."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data should be a Pandas DataFrame")
        
        self.data = data
    
    def pearson_correlation(self):
        """Computes Pearson correlation for selected columns."""
        return self.data.corr(method='pearson')
    
    def spearman_correlation(self):
        """Computes Spearman correlation for selected columns."""
        return self.data.corr(method='spearman')
    
    def kendall_correlation(self):
        """Computes Kendall correlation for selected columns."""
        return self.data.corr(method='kendall')
    
    def summary(self):
        """Returns a dictionary containing all correlation matrices."""
        return {
            "Pearson Correlation": self.pearson_correlation(),
            "Spearman Correlation": self.spearman_correlation(),
            "Kendall Correlation": self.kendall_correlation()
        }

# Example Usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        "C": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    })
    
    corr_obj = CorrelationTesting(df, ["A", "B", "C"])
    print(corr_obj.summary())
