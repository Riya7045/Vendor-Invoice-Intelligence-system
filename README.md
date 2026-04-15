# 📦 Vendor Invoice Intelligence System

An AI-powered machine learning application for predicting freight costs and detecting risky vendor invoices to optimize financial operations and reduce procurement fraud.

---

## 🎯 Project Overview

This system leverages advanced machine learning models to:
- **Forecast freight costs** with 96.99% accuracy using regression algorithms
- **Detect anomalous invoices** flagged for manual review to reduce financial risk
- **Streamline procurement decisions** through an interactive web interface

---

## 🎯 Business Objectives

- **Reduce Financial Leakage:** Minimize fraud and anomalies in vendor invoices through automated detection
- **Optimize Cost Forecasting:** Enable accurate freight cost predictions for better budget allocation
- **Accelerate Procurement Cycle:** Reduce manual review time by 40% through automated risk assessment
- **Improve Decision-Making:** Provide real-time insights for faster, data-driven vendor management decisions
- **Enhance Audit Efficiency:** Automate invoice screening for compliance and financial controls
- **Increase Vendor Accountability:** Identify patterns and outliers in vendor billing practices

---

## ✨ Key Features

- 🚚 **Freight Cost Prediction** - Predicts shipping costs using Linear Regression, Decision Tree, and Random Forest models
- 🚨 **Invoice Risk Flagging** - Identifies high-risk invoices using classification algorithms
- 📊 **Interactive Dashboard** - Real-time predictions via Streamlit web interface
- 💾 **Pre-trained Models** - Optimized models saved and ready for inference
- 🔄 **GitHub Integration** - Version control and easy deployment

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.14+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Web Framework** | Streamlit |
| **Visualization** | Plotly |
| **Model Serialization** | Joblib |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```
Vendor-Invoice-Intelligence-System/
├── app.py                                 # Main Streamlit application
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
│
├── freight_cost_prediction/               # Freight cost prediction module
│   ├── train.py                          # Model training script
│   ├── data_preprocessing.py             # Data processing utilities
│   └── model_evaluation.py               # Model evaluation metrics
│
├── invoice_flagging/                      # Invoice flagging module
│   ├── train.py                          # Model training script
│   ├── data_preprocessing.py             # Data processing utilities
│   └── modeling_evaluation.py            # Model evaluation metrics
│
├── inference/                             # Inference module
│   ├── predict_freight.py                # Freight prediction inference
│   ├── predict_invoice_flag.py           # Invoice flagging inference
│   └── __init__.py
│
├── models/                                # Trained ML models
│   ├── predict_freight_model.pkl         # Freight cost prediction model
│   ├── predict_flag_invoice.pkl          # Invoice flagging model
│   └── scaler.pkl                        # Feature scaler
│
├── notebooks/                             # Jupyter notebooks
│   ├── Predicting Freight Cost.ipynb     # Freight cost analysis
│   └── Invoice Flagging.ipynb            # Invoice flagging analysis
│
└── data/                                  # Database
    └── inventory.db                       # SQLite database
```

---

## 🚀 Installation

### Prerequisites
- Python 3.14 or higher
- pip or conda package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Riya7045/Vendor-Invoice-Intelligence-system.git
   cd Vendor-Invoice-Intelligence-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Model Performance

### Freight Cost Prediction
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** | 24.11 | 124.72 | **96.99%** ✅ |
| Decision Tree | 32.97 | 150.31 | 95.63% |
| Random Forest | 26.13 | 134.79 | 96.48% |

**Best Model:** Linear Regression (Lowest MAE, Highest R²)

### Invoice Flagging Classification
- **Algorithm:** Random Forest Classifier
- **Features Used:** 5 (invoice quantity, dollars, freight, total item quantity, total item dollars)
- **Outcome:** Accuracy in identifying risky invoices for manual review

---

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### Using the Streamlit Dashboard

**Freight Cost Prediction:**
1. Enter Invoice Quantity and Dollars
2. Click "Predict Freight Cost"
3. View estimated freight cost prediction

**Invoice Flagging:**
1. Enter invoice details (quantity, dollars, freight, etc.)
2. Click "Evaluate Invoice Risk"
3. See if invoice is flagged for manual review

---

## 🔧 Model Training

To retrain models with new data:

```bash
# Freight cost prediction
cd freight_cost_prediction
python train.py

# Invoice flagging
cd ../invoice_flagging
python train.py
```

---

## 📈 Sample Predictions

### Freight Cost Example
- Input: Invoice Dollars = $18,500
- Output: Estimated Freight = **$95.42**

### Invoice Flagging Example
- Input: Invoice Dollars = $352.95, Freight = $1.73, Total Items = 162
- Output: **✅ Safe for Auto-Approval** (No Flag)

---

## 📓 Jupyter Notebooks

Explore detailed analysis and visualizations:
- `notebooks/Predicting Freight Cost.ipynb` - Freight cost model development
- `notebooks/Invoice Flagging.ipynb` - Invoice flagging model analysis

---

## 🎓 Skills Demonstrated

- Machine Learning (Regression & Classification)
- Data Preprocessing & Feature Engineering
- Model Evaluation & Optimization
- Python Development
- Full-Stack Web Application Development
- GitHub & Version Control

---

## 🚀 Future Enhancements

- [ ] Deploy on Streamlit Cloud
- [ ] Add more vendor categories
- [ ] Implement time-series forecasting
- [ ] Create API endpoints for integration
- [ ] Add email alerts for flagged invoices
- [ ] Implement user authentication

---

## 👤 Author

**Riya**  
GitHub: [@Riya7045](https://github.com/Riya7045)  
Email: criya.07205@gmail.com

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

For questions or issues, please:
- Create an issue on [GitHub Issues](https://github.com/Riya7045/Vendor-Invoice-Intelligence-system/issues)
- Contact via email

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Last Updated:** April 2026  
**Status:** ✅ Active Development
