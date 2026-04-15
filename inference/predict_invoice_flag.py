import joblib
import pandas as pd
from pathlib import Path

def predict_invoice_flag(new_data, model_path=None, scaler_path=None):
    """
    Predict whether invoice should be flagged for risk.
    
    Parameters:
    -----------
    new_data : pd.DataFrame or dict
        New data with required features:
        ['invoice_quantity', 'invoice_dollars', 'Freight', 
         'total_item_quantity', 'total_item_dollars']
        
    model_path : str, optional
        Path to the trained model. Defaults to models/predict_flag_invoice.pkl
    
    scaler_path : str, optional
        Path to the scaler. Defaults to models/scaler.pkl
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with input data and predictions
        (0 = No flag, 1 = Flag for review)
    """
    
    REQUIRED_FEATURES = [
        'invoice_quantity', 'invoice_dollars', 'Freight',
        'total_item_quantity', 'total_item_dollars'
    ]
    
    # Default paths
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "predict_flag_invoice.pkl"
    
    if scaler_path is None:
        scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Convert dict to DataFrame if necessary
    if isinstance(new_data, dict):
        new_data = pd.DataFrame(new_data)
    
    # Ensure it's a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("new_data must be a DataFrame or dictionary")
    
    # Check for required features
    missing_features = [f for f in REQUIRED_FEATURES if f not in new_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select features and scale
    X = new_data[REQUIRED_FEATURES]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_probs = model.predict_proba(X_scaled)
    
    # Create result DataFrame
    result = new_data.copy()
    result["flag_prediction"] = predictions
    result["flag_probability"] = prediction_probs[:, 1]  # Probability of being flagged
    result["risk_status"] = result["flag_prediction"].apply(
        lambda x: "FLAG FOR REVIEW" if x == 1 else "NO FLAG"
    )
    
    return result


if __name__ == "__main__":
    # Example usage
    test_data = pd.DataFrame({
        "invoice_quantity": [50, 100, 75, 120, 60],
        "invoice_dollars": [500, 1000, 750, 1200, 600],
        "Freight": [50, 100, 75, 120, 60],
        "total_item_quantity": [500, 1000, 750, 1200, 600],
        "total_item_dollars": [4900, 9800, 7300, 11700, 5800]
    })
    
    results = predict_invoice_flag(test_data)
    print("\nInvoice Flag Predictions:")
    print(results[["invoice_dollars", "flag_prediction", "flag_probability", "risk_status"]])
    print(f"\nTotal invoices flagged: {results['flag_prediction'].sum()}")
