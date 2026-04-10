# Stock Price Movement Prediction Using Financial News Sentiment Analysis and Machine Learning

**7150CEM — Data Science Project | Coventry University**  
**Student:** Venkata Kiran Kumar Mullapudi (16200362)  
**Supervisor:** Dr. Rabia Ameen  
**Ethics Reference:** P193120  

---

## Project Overview

This project investigates whether integrating financial news sentiment analysis with machine learning models improves the accuracy of binary stock price movement classification for US equity markets.

**Prediction task:** Binary next-day directional classification — will the closing price go UP (1) or DOWN (0)?  
**Market:** 10 US tickers — AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, NFLX, ^GSPC  
**Period:** 2019–2023 (five years, 1,258 trading days per ticker)

---

## Repository Structure

| Notebook | Description |
|----------|-------------|
| `01-data-collection-eda.ipynb` | Downloads 12,560 OHLCV price rows via yfinance. Constructs binary direction label. Generates 5 EDA figures. |
| `02-sentiment-scoring.ipynb` | Scores 25,689 headlines using FinBERT (Tesla T4 GPU, 144.7s) and VADER (1.2s CPU). Produces 4,475 daily sentiment rows. |
| `03-feature-engineering.ipynb` | Merges price + sentiment. Computes 15 technical indicators. Builds 4 ablation feature matrices (15–28 features). Applies 70/15/15 time-ordered split. |
| `04-classical-ml.ipynb` | Trains Logistic Regression, SVM (grid search), and Random Forest across all 4 feature configurations. Generates results tables and figures. |
| `05-deep-learning.ipynb` | Trains LSTM and CNN-LSTM across all 4 configurations on GPU. Runs McNemar significance tests. Generates full ablation heatmap and ROC curves. |

---

## Key Results

| Model | Config | Accuracy | F1 | AUC-ROC |
|-------|--------|----------|----|---------|
| **Random Forest** | price_only | 51.16% | 56.65% | **51.62%** ← best overall |
| LSTM | price_finbert | 52.90% | 65.66% | 49.42% |
| CNN-LSTM | price_both | 53.02% | 64.49% | 49.49% |
| SVM | all configs | 54.18% | 70.28% | ~50.1% (degenerate) |
| Logistic Regression | price_only | 53.60% | 68.85% | 49.38% |

**Key finding:** Adding sentiment features consistently reduced performance across all five models (−0.29 to −2.38 pp). McNemar's test confirmed no statistically significant difference between models (p = 0.43) or ablation configurations (p = 0.78). The best discriminator was Random Forest on price-only technical features (AUC-ROC = 51.62%).

---

## Ablation Configurations

| Config | Features | Description |
|--------|----------|-------------|
| `price_only` | 15 | Technical indicators only (SMA, RSI, MACD, Bollinger, volatility) |
| `price_vader` | 20 | + 5 VADER sentiment features |
| `price_finbert` | 23 | + 8 FinBERT sentiment features |
| `price_both` | 28 | + 13 combined FinBERT + VADER features |

---

## Tools and Environment

- **Platform:** Kaggle (Jupyter Notebooks) — Tesla T4 GPU for NB2 and NB5
- **Language:** Python 3.12
- **Key libraries:** PyTorch, HuggingFace Transformers (FinBERT), scikit-learn, yfinance, pandas, statsmodels, matplotlib, seaborn
- **Sentiment models:** ProsusAI/finbert (domain-specific transformer), VADER (lexicon-based)
- **Statistical testing:** McNemar's test with Yates' continuity correction

---

## How to Run

1. Open each notebook on [Kaggle](https://www.kaggle.com) in order (NB1 → NB2 → NB3 → NB4 → NB5)
2. For NB2 and NB5: enable **GPU T4 x2** accelerator
3. For NB3, NB4: CPU is sufficient
4. Link each notebook's output to the next notebook via **+ Add Data → Notebook Outputs**
5. NB2 requires the Kaggle datasets: `miguelaenlle/massive-stock-news-analysis-db` and `bogomolov/daily-financial-news`

---

## Ethics

This project received ethical approval from Coventry University under reference **P193120**. All data was collected from publicly available sources. No personal data was collected or processed. Results are presented for academic purposes only and do not constitute financial or investment advice.

---

## Citation

If referencing this work:

> V. K. K. Mullapudi, "Stock Price Movement Prediction Using Financial News Sentiment Analysis and Machine Learning," MSc Data Science Project, Coventry University, 2026.
