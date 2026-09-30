# E-Commerce Customer Segmentation

A simple machine-learning project for PS-06: E-Commerce Customer Segmentation.

## Objective
Group customers according to purchasing behavior using:
- Total Spending
- Number of Purchases
- Purchase Frequency
- Average Order Value
- Recency

## ML methods
1. K-Means Clustering (main model)
2. Agglomerative Clustering (comparison)

## Dashboard
The Streamlit dashboard:
- loads the trained K-Means model
- shows customer segment distribution
- displays segment profiles
- allows a user to enter customer behavior and get a segment prediction.

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app/app.py
```

The project includes a self-contained sample dataset in `data/customer_data.csv`.
