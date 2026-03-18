"""ML pipeline for swing-trade forecasting.

Modules:
- features: extract feature vectors from price/indicator data
- labels: triple-barrier labeling for supervised learning
- trainer: walk-forward LightGBM training with purged validation
- scorer: load a trained model and score new setups
"""
