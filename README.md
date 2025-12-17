# Player Minutes Prediction for EFL League One

## Overview

This project predicts **player minutes played** in upcoming matches for **EFL League One football teams** using historical match-level and season-level performance data. The aim is to support coaching staff and analysts with data-driven insights into squad utilisation, rotation, and workload management.

The project is developed as part of an **MSc in Artificial Intelligence** and follows **industry-standard machine learning practices**, with a clear separation between code and raw data.

---

## Objectives

- Predict minutes played per player per match
- Engineer meaningful player and team features from historical data
- Compare baseline and advanced machine learning models
- Provide interpretable results suitable for football analytics use cases
- Deploy an interactive **Streamlit dashboard** for exploration

---

## Project Structure

```
player-minutes-prediction-efl-league-one/
│
├── model2.ipynb          # Main analysis & modelling notebook
├── streamlit_app.py      # Streamlit application
├── .gitignore            # Git ignore rules (raw data excluded)
├── outputs/              # Generated outputs (plots, tables)
└── README.md             # Project documentation
```

---

## Data

⚠️ **Raw datasets are not included in this repository** due to GitHub file size limits and best practices for machine learning projects.

### Data characteristics

- Match-level player statistics (minutes, actions, performance metrics)
- Team and opponent context
- Season-level aggregates
- Multiple seasons (2019–present)

### Data access

The datasets are:

- Stored locally for development
- Provided by **StatsBomb** and club data sources
- Available on request for academic review

If you wish to run the code locally, update the file paths in the notebook to point to your local data directory.

---

## Methodology

The project follows a structured ML workflow:

1. **Data auditing & schema validation**
2. **Pre-processing & cleaning**

   - Missing value handling
   - Feature normalisation

3. **Feature engineering**

   - Player form and workload features
   - Team-level context features

4. **Modelling**

   - Baseline models
   - Tree-based and polynomial models
   - Dimensionality reduction (PCA / Isomap)
   - Clustering for feature enrichment

5. **Evaluation**

   - Error metrics
   - Residual analysis

6. **Visualisation & deployment**

---

## Streamlit Application

The Streamlit app allows users to:

- Select players and seasons
- Visualise predicted vs actual minutes
- Explore feature effects interactively

To run locally:

```bash
streamlit run streamlit_app.py
```

---

## Requirements

Core dependencies:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- streamlit

(See notebook for full import list.)

---

## Academic Context

- Degree: **MSc Artificial Intelligence**
- Project theme: Artificial Intelligence Systems
- Focus: Sports analytics & predictive modelling

This repository is structured to meet **academic integrity requirements** and **industry best practices**.

---

## Author

**Rahmath Mozumder**
Software & AI Engineer
MSc Artificial Intelligence

---

## Licence

This project is for **academic and demonstration purposes**. Commercial use of underlying football data may be subject to data provider restrictions.
