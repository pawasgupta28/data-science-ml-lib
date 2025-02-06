import pandas as pd

class MissingDataAnalysis:
    def __init__(self, data):
        """Initialize with a Pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data should be a Pandas DataFrame")
        
        self.data = data
    
    def missing_summary(self):
        """Returns a summary of missing values for each column."""
        return self.data.isnull().sum().to_dict()
    
    def missing_percentage(self):
        """Returns the percentage of missing values for each column."""
        return (self.data.isnull().mean() * 100).to_dict()
    
    def missing_by_dtype(self):
        """Returns missing value counts categorized by data type (numeric and categorical)."""
        numeric_missing = self.data.select_dtypes(include=['number']).isnull().sum()
        categorical_missing = self.data.select_dtypes(exclude=['number']).isnull().sum()
        return {
            "Numeric Columns": numeric_missing.to_dict(),
            "Categorical Columns": categorical_missing.to_dict()
        }
    
    def summary(self):
        """Returns a dictionary summarizing missing data analysis."""
        return {
            "Missing Summary": self.missing_summary(),
            "Missing Percentage": self.missing_percentage(),
            "Missing by Data Type": self.missing_by_dtype()
        }

# Example Usage
if __name__ == "__main__":
    df = pd.DataFrame({
        "A": [1, 2, None, 4, 5],
        "B": ["a", None, "c", "d", "e"],
        "C": [None, None, 3, 4, 5]
    })
    
    missing_data = MissingDataAnalysis(df)
    print(missing_data.summary())
