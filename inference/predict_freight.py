import joblib
import pandas as pd
from pathlib import Path

def predict_freight_cost(new_data, model_path=None):
    """
    Predict freight cost for new vendor invoice data.
    
    Parameters:
    -----------
    new_data : pd.DataFrame or dict
        New data with 'Dollars' column/key
        Example: {'Dollars': [100, 200, 150]}
        or pd.DataFrame with columns ['Dollars']
    
    model_path : str, optional
        Path to the trained model. Defaults to models/predict_freight_model.pkl
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with input data and predictions
    """
    
    # Default model path
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "predict_freight_model.pkl"
    
    # Load the model
    model = joblib.load(model_path)
    
    # Convert dict to DataFrame if necessary
    if isinstance(new_data, dict):
        new_data = pd.DataFrame(new_data)
    
    # Ensure it's a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("new_data must be a DataFrame or dictionary")
    
    # Select only 'Dollars' column
    X = new_data[["Dollars"]]
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create result DataFrame
    result = new_data.copy()
    result["predicted_freight"] = predictions
    
    return result


if __name__ == "__main__":
    # Example usage
    test_data = pd.DataFrame({
        "Dollars": [100, 200, 150, 300, 250]
    })
    
    results = predict_freight_cost(test_data)
    print("\nFreight Cost Predictions:")
    print(results)
    print(f"\nMean predicted freight: ${results['predicted_freight'].mean():.2f}")
