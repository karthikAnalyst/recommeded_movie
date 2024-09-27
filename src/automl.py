import warnings
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits  # Placeholder dataset

# Ignore torch warning if PyTorch is not installed
warnings.filterwarnings('ignore', category=UserWarning, message="Warning: optional dependency `torch` is not available.")

# Example dataset (replace with your actual data)
digits = load_digits()
X, y = digits.data, digits.target

def select_best_model(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TPOT for automated model selection
    tpot = TPOTClassifier(verbosity=2, generations=5)
    tpot.fit(X_train, y_train)
    
    # Print the test accuracy
    print("Test Accuracy: ", tpot.score(X_test, y_test))
    
    # Export the best pipeline
    tpot.export('best_model_pipeline.py')

# Call function with your data
select_best_model(X, y)
