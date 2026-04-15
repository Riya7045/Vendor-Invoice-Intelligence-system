"""
Unified inference module for freight cost prediction and invoice flagging.
Allows batch predictions on new test data using both trained models.
"""

import joblib
import pandas as pd
from pathlib import Path
from predict_freight import predict_freight_cost
from predict_invoice_flag import predict_invoice_flag


def predict_all(new_data, models_dir=None):
    """
    Make predictions for both freight cost and invoice flagging.
    
    Parameters:
    -----------
    new_data : pd.DataFrame or dict
        New data with all required features:
        - For freight: 'Dollars'
        - For invoice flag: 'invoice_quantity', 'invoice_dollars', 'Freight',
                          'total_item_quantity', 'total_item_dollars'
    
    models_dir : str, optional
        Path to models directory. Defaults to ../models
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all predictions
    """
    
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    else:
        models_dir = Path(models_dir)
    
    # Convert dict to DataFrame if necessary
    if isinstance(new_data, dict):
        new_data = pd.DataFrame(new_data)
    
    # Get freight predictions
    freight_results = predict_freight_cost(
        new_data, 
        model_path=str(models_dir / "predict_freight_model.pkl")
    )
    
    # Get invoice flag predictions
    invoice_results = predict_invoice_flag(
        new_data,
        model_path=str(models_dir / "predict_flag_invoice.pkl"),
        scaler_path=str(models_dir / "scaler.pkl")
    )
    
    # Combine results
    combined = new_data.copy()
    combined["predicted_freight"] = freight_results["predicted_freight"]
    combined["flag_prediction"] = invoice_results["flag_prediction"]
    combined["flag_probability"] = invoice_results["flag_probability"]
    combined["risk_status"] = invoice_results["risk_status"]
    
    return combined


if __name__ == "__main__":
    # Example: Full workflow with multiple test records
    test_data = pd.DataFrame({
        "Dollars": [500, 1000, 750, 1200, 600],
        "invoice_quantity": [50, 100, 75, 120, 60],
        "invoice_dollars": [500, 1000, 750, 1200, 600],
        "Freight": [50, 100, 75, 120, 60],
        "total_item_quantity": [500, 1000, 750, 1200, 600],
        "total_item_dollars": [4900, 9800, 7300, 11700, 5800]
    })
    
    print("=" * 80)
    print("UNIFIED INFERENCE RESULTS")
    print("=" * 80)
    
    results = predict_all(test_data)
    
    print("\nFull Results:")
    print(results)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average predicted freight: ${results['predicted_freight'].mean():.2f}")
    print(f"Total invoices flagged: {results['flag_prediction'].sum()} out of {len(results)}")
    print(f"Average flag probability: {results['flag_probability'].mean():.2%}")
    
    print("\nDetailed Results by Invoice:")
    for idx, row in results.iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"  Invoice Dollars: ${row['invoice_dollars']:.2f}")
        print(f"  Predicted Freight: ${row['predicted_freight']:.2f}")
        print(f"  Risk Status: {row['risk_status']} (probability: {row['flag_probability']:.2%})")
