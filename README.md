# Stock Price Movement Prediction Using Financial News Sentiment Analysis and Machine Learning

**Module:** 7150CEM — Data Science Project | Coventry University  
**Student:** Venkata Kiran Kumar Mullapudi · ID: 16200362  
**Supervisor:** Dr. Rabia Ameen · **Ethics:** P193120  

---

## What This Project Does

This project answers one question: **does adding financial news sentiment to machine learning models help predict whether a stock price will go up or down the next day?**

Five years of daily price data (2019–2023) for ten major US stocks were collected. 25,689 financial news headlines were scored using two sentiment tools — FinBERT (a domain-specific transformer model) and VADER (a lightweight lexicon tool). Five machine learning models were then trained and evaluated across four different feature combinations to measure exactly how much sentiment features add — or do not add — to prediction accuracy.

---

## Dataset

| Data | Detail |
|------|--------|
| Price data | 10 tickers · 1,258 trading days each · 12,560 rows total · zero missing values |
| Tickers | AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, NFLX, ^GSPC |
| Period | 1 Jan 2019 – 31 Dec 2023 |
| Headlines | 25,689 unique headlines from Yahoo Finance, Reuters, Benzinga and Reddit WallStreetBets |
| Sentiment coverage | 29.7% of trading days had real news · remaining days forward-filled |
| Label | 1 = price goes up next day · 0 = price goes down |

---

## Pipeline — 5 Notebooks

```
NB1 → NB2 → NB3 → NB4 → NB5
```

| Notebook | What it does |
|----------|-------------|
| `01-data-collection-eda` | Downloads OHLCV price data via yfinance. Builds binary direction label. Loads and filters 4.7M raw headlines to 25,689 matched entries. |
| `02-sentiment-scoring` | Runs FinBERT on GPU (Tesla T4 · 144.7s · 177 headlines/sec). Runs VADER on CPU (1.2s). Produces 4,475 daily sentiment rows with 10 features per ticker. |
| `03-feature-engineering` | Calculates 15 technical indicators (SMA, RSI, MACD, Bollinger Bands, volatility). Merges price + sentiment. Builds 4 ablation feature matrices. Applies 70/15/15 chronological split. |
| `04-classical-ml` | Trains Logistic Regression, SVM (grid search), and Random Forest. Evaluates all 4 feature configurations. Generates ablation bar chart and feature importance plot. |
| `05-deep-learning` | Trains LSTM and CNN-LSTM on GPU. Runs McNemar significance tests. Generates full 5×4 ablation heatmap and ROC curves. |

**Run order:** NB1 → NB2 → NB3 → NB4 → NB5. Link each notebook's Kaggle output to the next via **+ Add Data → Notebook Outputs**. GPU required for NB2 and NB5 only.

---

## Models and Configurations

**4 feature configurations (ablation study):**

| Config | Features | What's included |
|--------|----------|----------------|
| `price_only` | 15 | Technical indicators only |
| `price_vader` | 20 | + 5 VADER daily sentiment features |
| `price_finbert` | 23 | + 8 FinBERT daily sentiment features |
| `price_both` | 28 | + all 13 FinBERT + VADER features |

**5 models evaluated across all 4 configurations = 20 experiments total:**
- Logistic Regression (linear baseline)
- SVM with RBF kernel (grid search: C, gamma)
- Random Forest (200 trees, Gini feature importance)
- LSTM (2-layer · 128→64 units · dropout 0.3 · 20-day lookback)
- CNN-LSTM hybrid (dual-branch · local pattern extraction + sequential modelling)

---

## Results

**Best model overall: Random Forest · price_only · AUC-ROC = 51.62%**

| Model | Config | Accuracy | F1 | AUC-ROC |
|-------|--------|----------|----|---------|
| Random Forest | price_only | 51.16% | 56.65% | **51.62%** |
| CNN-LSTM | price_both | 53.02% | 64.49% | 49.49% |
| LSTM | price_finbert | 52.90% | 65.66% | 49.42% |
| SVM | all configs | 54.18% | 70.28% | ~50.1%* |
| Logistic Regression | price_only | 53.60% | 68.85% | 49.38% |

*SVM predicted "Up" for nearly every sample (Recall ≈ 100%) — degenerate classifier.

**Ablation finding:** Sentiment features reduced performance in every model.

| Model | price_only → price_both accuracy change |
|-------|-----------------------------------------|
| Logistic Regression | −2.38 pp |
| CNN-LSTM | −2.07 pp |
| Random Forest | −1.53 pp |
| LSTM | −0.29 pp |
| SVM | 0.00 pp (already degenerate) |

**Statistical significance:** McNemar's test confirmed no significant difference between LSTM and CNN-LSTM (p = 0.43) or between ablation configurations (p = 0.78).

---

## Key Finding

> Sentiment features did not improve stock price direction prediction in this study. All models performed close to chance (AUC-ROC 47–52%). The result is consistent with the Efficient Market Hypothesis and the 29.7% news coverage limitation — 70.3% of days relied on forward-filled sentiment values rather than real news signal.

This is a rigorous null result. It establishes that **daily aggregated sentiment at this coverage level does not add discriminative value** to price-based technical indicators for next-day binary classification.

---

## Tech Stack

| Tool | Use |
|------|-----|
| Python 3.12 | All notebooks |
| PyTorch 2.x | LSTM and CNN-LSTM training |
| HuggingFace Transformers | FinBERT (ProsusAI/finbert) |
| scikit-learn 1.4 | Classical ML models and StandardScaler |
| yfinance | OHLCV price data collection |
| statsmodels | McNemar's significance test |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | All figures |
| Kaggle (Tesla T4 GPU) | NB2 and NB5 execution |

---

## Ethics

Ethical approval: Coventry University reference **P193120**.  
All data collected from publicly available sources under open licences.  
No personal data collected or processed.  
Results are academic outputs only — not financial or investment advice.

---

## How to Cite

> V. K. K. Mullapudi, "Stock Price Movement Prediction Using Financial News Sentiment Analysis and Machine Learning," MSc Data Science Project, Coventry University, 2026. Available: https://github.com/mullapud3/7150CEM-Data-Science-Project
