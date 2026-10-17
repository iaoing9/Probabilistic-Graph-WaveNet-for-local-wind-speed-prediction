# Probabilistic-Graph-WaveNet-for-local-wind-speed-prediction

Source code and prediction results for a probabilistic Graph WaveNet for local wind speed prediction combining meteorological data and on-site measurements

Case 1: Retrospective hindcasting of mean wind speed using ECMWF reanalysis data and on-site measurements from the long-span bridge site.
Case 2: Operational forecasting of wind gust using ECMWF forecasts and on-site measurements from two anemometric stations.

Brief Overview and Notes:

1. Model naming conventions: GWN stands for Graph WaveNet; TM for Transformer; LSTM for long short-term memory; GRU for gated recurrent unit; QRF for quantile regression forest; SGPR for sparse Gaussian process regression; PM for persistence model; ARIMA for autoregressive integrated moving average; and VAR for vector autoregression. MSE and graph off denote the loss function and graph component ablation variants, respectively. ECMWF only and ANE only denote alternative input configurations using only ECMWF meteorological data or ANE on-site measurements, respectively. The prefix “P-” denotes the probabilistic implementation of the corresponding model.

2. Prediction files: The timestamp-aligned, sample-level predictions for the two cases are provided in two separate Excel workbooks, each containing four sheets: S1. Comparison with benchmarks; S2. Model ablation study; S3. Rolling-origin evaluation; and S4. Multi-seed evaluation. The measurements are subject to data-use restrictions and cannot be publicly shared or redistributed. The PM predictions are also excluded from the released files as they could disclose the underlying measurements.

3. Probabilistic prediction diagnostics: The diagnostic scripts include Standardized residual diagnostics.py and Distributional calibration diagnostics.py. These scripts are designed as post-processing tools independent of the model implementations. They operate on the generated predictive outputs to assess probabilistic prediction performance.

4. Evaluation scripts: The scripts include Metric evaluation-Gaussian.py and Metric evaluation-Quantile.py. The former is designed to calculate evaluation metrics and scores for all models except QRF, while the latter is designed specifically for QRF.

5. DM test: In DM test.py, SE and CRPS are adopted as two separate loss functions to comprehensively evaluate the probabilistic prediction performance.






