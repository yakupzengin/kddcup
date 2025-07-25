# KDD Cup 2022 ESCI Challenge: Shopping Queries Dataset

This project implements a comprehensive solution for the KDD Cup 2022 ESCI (Exact, Substitute, Complement, Irrelevant) Challenge using the Shopping Queries Dataset.

## 🎯 Challenge Overview

The ESCI Challenge focuses on improving product search through three tasks:

- **Task 1**: Query-Product Ranking - Rank products for relevance to search queries
- **Task 2**: Multi-class Product Classification - Classify products as E/S/C/I 
- **Task 3**: Product Substitute Identification - Identify substitute products

## 📁 Project Structure

```
esci_challenge/
├── data/
│   ├── raw/                          # Original dataset files
│   ├── processed/                    # Cleaned data by task
│   └── features/                     # Generated features
├── src/
│   ├── config/                       # Configuration files
│   ├── data/                         # Data loading utilities
│   ├── features/                     # Feature engineering
│   ├── models/                       # Model implementations
│   ├── evaluation/                   # Evaluation metrics
│   └── utils/                        # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data understanding
│   ├── 02_feature_engineering.ipynb # Feature development
│   ├── 03_baseline_model.ipynb      # Baseline implementation
│   └── 04_model_improvement.ipynb   # Advanced models
├── scripts/
│   ├── train_task1.py               # Task 1 training
│   ├── train_task2.py               # Task 2 training (TODO)
│   └── train_task3.py               # Task 3 training (TODO)
├── results/
│   ├── models/                      # Trained models
│   ├── predictions/                 # Model predictions
│   └── evaluation/                  # Performance metrics
└── experiments/                     # Experiment tracking
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup project structure and check data
python main.py --setup-only
```

### 2. Download Data

Download the KDD Cup 2022 dataset files and place them in `data/raw/`:
- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`  
- `shopping_queries_dataset_sources.csv`

### 3. Run Training Pipeline

```bash
# Run Task 1 (Query-Product Ranking) - Baseline
python main.py --task 1

# Run all tasks
python main.py --all-tasks

# Run with custom experiment name
python main.py --task 1 --experiment baseline_v2
```

## 📊 Development Roadmap

### Phase 1: Foundation (Current)
- ✅ Project structure setup
- ✅ Data exploration notebook
- ✅ Basic feature engineering
- ✅ Task 1 baseline implementation
- 🎯 **Target**: NDCG ~0.2-0.3

### Phase 2: Feature Enhancement
- 🔄 Advanced text features (TF-IDF, embeddings)
- 🔄 Query-product similarity metrics
- 🔄 Statistical features
- 🎯 **Target**: NDCG ~0.4-0.5

### Phase 3: Model Optimization
- ⏳ Hyperparameter tuning
- ⏳ Feature selection optimization
- ⏳ Ensemble methods
- 🎯 **Target**: NDCG ~0.6-0.7

### Phase 4: Advanced Techniques
- ⏳ Neural ranking models
- ⏳ Cross-validation strategies
- ⏳ Multi-task learning
- 🎯 **Target**: NDCG ~0.7+

## 🔧 Key Features

### Data Processing
- Automated data loading and preprocessing
- Task-specific data filtering (small/large versions)
- English-only filtering for focused development

### Feature Engineering
- **Text Features**: Length, word count, character statistics
- **Similarity Features**: Query-title cosine similarity, exact matches
- **Statistical Features**: Query frequency, product popularity
- **Advanced Features**: TF-IDF vectors, semantic embeddings (planned)

### Model Implementation
- **LightGBM Ranker**: Optimized for ranking tasks
- **Proper Validation**: Query-level train/val splits to prevent leakage
- **Evaluation**: NDCG metrics for ranking performance

### Experiment Tracking
- Organized result storage by experiment
- Model versioning and comparison
- Performance tracking across iterations

## 📈 Performance Tracking

| Experiment | Features | NDCG@10 | Notes |
|------------|----------|---------|-------|
| baseline_v1 | Basic text + similarity | TBD | Initial implementation |
| feature_v1 | + TF-IDF features | TBD | Enhanced text features |
| feature_v2 | + Statistical features | TBD | Query/product statistics |
| optimized_v1 | + Hyperparameter tuning | TBD | Model optimization |

## 🛠️ Usage Examples

### Running Experiments

```bash
# Baseline experiment
python main.py --task 1 --experiment baseline_v1

# Feature engineering experiment  
python main.py --task 1 --experiment feature_v1

# Check results
ls results/models/        # Trained models
ls results/predictions/   # Predictions
ls results/evaluation/    # Performance metrics
```

### Using Notebooks

1. **Data Exploration**: `notebooks/01_data_exploration.ipynb`
   - Dataset statistics and distributions
   - ESCI label analysis
   - Text characteristics

2. **Feature Engineering**: `notebooks/02_feature_engineering.ipynb`
   - Feature development and testing
   - Feature importance analysis
   - Correlation studies

3. **Model Development**: `notebooks/03_baseline_model.ipynb`
   - Baseline model implementation
   - Performance evaluation
   - Error analysis

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- scikit-learn >= 1.1.0
- lightgbm >= 3.3.0
- matplotlib, seaborn, plotly
- jupyter notebooks

## 🎯 Next Steps

1. **Run baseline model** to establish performance benchmark
2. **Explore data** using the provided notebooks
3. **Enhance features** with text similarity and statistical measures
4. **Optimize hyperparameters** for improved performance
5. **Implement Tasks 2 & 3** using similar methodology

## 📝 Notes

- Focus on **English queries only** (`product_locale == "us"`)
- Use **small_version == 1** for Task 1 development
- Implement **proper ranking evaluation** with query-level grouping
- Track experiments systematically for reproducible results

## 🤝 Contributing

1. Create feature branches for new developments
2. Use descriptive experiment names
3. Document performance improvements
4. Follow the established project structure

---

**Happy Coding! 🚀**

*Let's build an awesome ranking system for e-commerce search!*#   k d d c u p 
 
 