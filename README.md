
⚡ Load Forecasting and Demand Response Optimization in Microgrids using Machine Learning

This repository presents a comprehensive project on short-term load forecasting and demand response (DR) optimization in microgrids using LSTM neural networks and linear programming. The solution includes time-series forecasting using deep learning, followed by optimization of electricity consumption to reduce peak demand.

📁 Project Structure

project_files

├── AEP_hourly.csv                # Historical hourly load data from AEP

├── code.py                       # Main Python script with model training and optimization

├── code1.ipynb                   # Jupyter Notebook version for interactive analysis

├── models

│      └── lstm_load_forecast.h5     # Trained LSTM model file

├── plots

│      ├── load_forecast.png         # Actual vs Predicted load plot

│      ├── error_distribution.png    # Forecasting error distribution

│      └── microgrid_load_dr.png     # Load profile before and after DR

├── predictions

│      ├── predictions.csv              # Model predictions vs actual values

│      ├── microgrid_load_profile.csv   # Original 24-hour microgrid load profile

│      └── microgrid_load_with_dr.csv  # Load adjusted after DR optimization

└── reports

│      ├── load_forecasting_report.txt # Evaluation metrics for the LSTM model

│      └── final_project_report.txt     # Summary of forecasting and DR results

└── README.md

└── README.md

🎯 Project Objectives

- Accurately forecast short-term electrical load using historical hourly data.
- Apply LSTM (Long Short-Term Memory) neural networks for time series forecasting.
- Evaluate forecasting performance using MAE, RMSE, MAPE, and R².
- Implement demand response optimization using linear programming to reduce peak load.
- Visualize the impact of DR on microgrid consumption.

📊 Model Performance

| Metric  | Value       |
|---------|-------------|
| MAE     | 154.87 MW   |
| RMSE    | 202.16 MW   |
| MAPE    | 1.06%       |
| R² Score| 0.9932      |

> The model demonstrates high predictive accuracy and low error rates, making it suitable for real-world microgrid applications.

⚙️ How to Run

1. Clone the repository

2. Install dependencies (if needed):
   pip install -r requirements.txt

3. Run the main script:
   python code.py

   Or explore step-by-step via Jupyter Notebook:
   jupyter notebook code.ipynb

📈 Visualizations

- load_forecast.png: Comparison of actual and predicted load values over time.
- error_distribution.png: Histogram of prediction errors (Actual - Predicted).
- microgrid_load_dr.png: 24-hour load profile before and after DR optimization.

✅ Demand Response Optimization

- Original total load: Refer to microgrid_load_profile.csv
- Optimized load after DR: 275.75 kW (see microgrid_load_with_dr.csv)
- Optimization performed using linear programming with peak shaving objectives.

📌 Highlights

- End-to-end machine learning pipeline for energy forecasting.
- Integration of AI with operational energy optimization.
- Clean folder structure, modular code, and reproducible results.
- Ready for extension with new datasets or models.

📚 Reports

- load_forecasting_report.txt: Contains LSTM performance evaluation on test/train data.
- final_project_report.txt: Full summary of the project including forecasting and optimization steps.

🤝 Acknowledgments

This project was developed as part of an academic research initiative in smart energy systems and microgrids.

📬 Contact

For questions, feedback, or collaboration inquiries, feel free to reach out via GitHub Issues or email.
