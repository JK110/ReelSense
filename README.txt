# ReelSense: Explainable Movie Recommender System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens-orange.svg)
![Competition](https://img.shields.io/badge/Competition-BrainDead%202K26-red.svg)

**Submission for BrainDead Competition**  
**Revelation 2K26 | IIEST Shibpur**  
*Department of Computer Science and Technology*

---

### 🎯 An Explainable AI Approach to Movie Recommendations

*Combining Data Analysis, Machine Learning, and Natural Language Explanations*

</div>

---

## 🏆 Competition Information

**Event**: BrainDead - Data Analysis and Machine Learning Challenge  
**Organizer**: Department of Computer Science and Technology, IIEST Shibpur  
**Fest**: Revelation 2K26  
**Theme**: Data Analysis & Machine Learning Innovation  

This project addresses the challenge of building transparent, explainable recommendation systems that users can trust and understand.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results & Insights](#-results--insights)
- [Visualizations](#-visualizations)
- [Future Enhancements](#-future-enhancements)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Team & Acknowledgments](#-team--acknowledgments)
- [References](#-references)

---

## 🎯 Project Overview

**ReelSense** is an explainable movie recommendation system that goes beyond traditional black-box approaches. Built for the BrainDead competition at Revelation 2K26, this project demonstrates the power of combining data analysis, machine learning, and explainable AI to create user-centric recommendations.

### Why ReelSense?

In an era where recommendation systems drive user engagement across platforms, **transparency** and **trust** are paramount. ReelSense provides:

- 🔍 **Transparent Recommendations**: Clear explanations for every suggestion
- 📊 **Data-Driven Insights**: Comprehensive exploratory data analysis
- 🎭 **Diversity-Aware**: Balances popularity with catalog coverage
- 🤖 **Scalable Architecture**: Foundation for advanced ML models

---

## 💡 Problem Statement

**Challenge**: Develop a recommendation system that not only predicts user preferences accurately but also explains its reasoning in natural language.

**Key Requirements**:
1. ✅ Implement data preprocessing and feature engineering
2. ✅ Conduct thorough exploratory data analysis
3. ✅ Build and evaluate recommendation models
4. ✅ Provide interpretable explanations for recommendations
5. ✅ Measure diversity, coverage, and novelty metrics

---

## ✨ Features

### 🔑 Core Capabilities

- **Explainable Recommendations**: Natural language explanations linking suggestions to user history
- **Multi-Metric Evaluation**: Precision, Recall, NDCG, Coverage, Diversity, Novelty
- **Feature Engineering**: Advanced one-hot encoding with cosine similarity (1,496 features)
- **Temporal Splitting**: Realistic time-based train-test methodology
- **Visualization Suite**: Comprehensive EDA with distribution and trend analysis

### 🎨 Unique Selling Points

1. **User-Centric Explanations**: "Because you liked X and Y, both are Z genre films"
2. **Diversity Metrics**: Ensures recommendations aren't just popular blockbusters
3. **Scalable Design**: Baseline for implementing collaborative filtering, matrix factorization
4. **Reproducible Research**: Detailed documentation and modular code structure

---

## 📊 Dataset

**Source**: [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/)

### Dataset Composition

| Component | Description | Size |
|-----------|-------------|------|
| **Ratings** | User ratings (0.5-5.0 scale) | 100,836 ratings |
| **Movies** | Movie metadata | 9,742 movies |
| **Tags** | User-generated tags | 3,683 tags |
| **Users** | Unique users | 610 users |

### Processed Features

```
📁 Dataset Statistics
├── Training Set: 100,226 ratings (610 users × 9,701 movies)
├── Test Set: 610 ratings (last rating per user)
├── Genre Features: 21 unique genres (one-hot encoded)
├── Tag Features: 1,476 unique tags (one-hot encoded)
└── Similarity Matrix: 9,742 × 9,742 (cosine similarity)
```

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
pip package manager
```

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/reelsense-braindead.git
cd reelsense-braindead

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MovieLens dataset
# Place in data/ directory or run:
python scripts/download_data.py

# 4. Launch Jupyter Notebook
jupyter notebook ReelSense.ipynb
```

### Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

---

## 💻 Usage

### Quick Start Example

```python
import pandas as pd
from reelsense import ReelSenseRecommender

# Load datasets
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
tags = pd.read_csv('data/tags.csv')

# Initialize and train
recommender = ReelSenseRecommender()
recommender.fit(ratings, movies, tags)

# Get top-10 recommendations
user_id = 42
recs = recommender.recommend(user_id, n=10)
print(recs)

# Generate explanation
movie_id = 318  # Shawshank Redemption
explanation = recommender.explain(user_id, movie_id)
print(explanation)
# Output: "We recommend 'Shawshank Redemption (1994)' because you 
#          liked Godfather, The (1972), Fight Club (1999) and are 
#          both 'Drama' films."
```

### Running the Full Analysis

```bash
# Execute complete pipeline in Jupyter
jupyter notebook ReelSense.ipynb

# Or run as Python script
python src/main.py --mode train --evaluate --explain
```

---

## 🔬 Methodology

### 1️⃣ Data Preprocessing Pipeline

```python
# Timestamp conversion
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Time-based split (Leave-Last-1)
train, test = temporal_split(ratings, n=1)

# Genre cleaning
movies['genres'] = movies['genres'].str.split('|')

# Tag normalization
tags['tag'] = tags['tag'].str.lower().str.strip()

# Feature matrix construction
movie_features = one_hot_encode(genres, tags)  # 9,742 × 1,496
similarity_matrix = cosine_similarity(movie_features)
```

### 2️⃣ Baseline Recommender

**Popularity-Based Approach**:
- Rank movies by average rating
- Minimum threshold: 50 ratings
- Non-personalized benchmark

**Why This Baseline?**:
- Establishes performance floor
- Tests evaluation framework
- Identifies improvement opportunities

### 3️⃣ Explainability Engine

**Natural Language Generation**:

```python
def generate_explanation(user_id, movie_id):
    # 1. Get user's top-rated movies
    user_favs = get_top_rated(user_id, n=3)
    
    # 2. Find shared genres/tags with recommended movie
    shared_features = find_common_features(user_favs, movie_id)
    
    # 3. Construct natural language explanation
    return f"Because you liked {user_favs} and are both {shared_features} films."
```

### 4️⃣ Evaluation Framework

**Ranking Metrics** (Top-K = 10):
- Precision@10: Relevance of recommendations
- Recall@10: Coverage of relevant items
- NDCG@10: Position-aware relevance
- Hit Ratio: Binary relevance indicator
- MAP: Mean Average Precision

**Diversity Metrics**:
- Catalog Coverage: % of catalog recommended
- Intra-List Diversity: Feature diversity within recommendations
- Popularity-Normalized Hits: Novelty measure

---

## 📈 Results & Insights

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision@10** | 0.0018 | Very low (non-personalized baseline) |
| **Recall@10** | 0.0180 | Limited relevant item coverage |
| **NDCG@10** | 0.0096 | Poor ranking quality |
| **Hit Ratio** | 0.0180 | 1.8% of users had ≥1 hit |
| **MAP** | 0.0180 | Low average precision |
| **Catalog Coverage** | 0.0010 | Only 0.1% of catalog used |
| **Intra-List Diversity** | 0.8079 | ✅ High diversity within top-10 |
| **Novelty Score** | 0.2069 | Low (expected for popularity) |

### Key Insights from EDA

#### 📊 Rating Distribution
- **Peak**: 4.0-5.0 stars
- **Insight**: Users rate movies they enjoy → positive bias in dataset
- **Implication**: Need to account for rating inflation in models

#### 🎬 Genre Analysis
**Most Popular** (by count):
1. Drama (25,606 ratings)
2. Comedy (16,870 ratings)
3. Action (13,234 ratings)

**Highest Rated** (by average):
1. Film-Noir (4.12)
2. Documentary (3.95)
3. War (3.89)

#### 👥 User Behavior
- **Long-tail distribution**: 80% of users provide <100 ratings
- **Power users**: Top 10% account for 45% of all ratings
- **Temporal trends**: Rating activity peaked in 2000, sustained since 2015

#### 🎭 Movie Popularity
- **Blockbuster effect**: Top 1% of movies receive 25% of ratings
- **Long tail**: 40% of movies have <10 ratings
- **Coverage challenge**: Recommender must balance popularity vs. discovery

---

## 📊 Visualizations

<details>
<summary><b>Click to view analysis visualizations</b></summary>

### Rating Distribution
```
5.0 ████████████████████████████ 28,750
4.5 ██████████████████ 18,245
4.0 ███████████████████████ 23,456
3.5 ████████████ 12,340
3.0 ██████████ 10,123
...
```

### Genre Popularity
```
Drama    ████████████████████████████ 25,606
Comedy   ███████████████████ 16,870
Action   ███████████████ 13,234
Thriller ████████████ 10,542
Romance  ██████████ 9,128
```

### User Activity Distribution
```
Most users: 20-50 ratings (Long-tail pattern)
Power users: 200+ ratings (Top 5%)
```

</details>

---

## 🔮 Future Enhancements

### Phase 1: Advanced Models (Next Steps)
- [ ] **Collaborative Filtering**: User-based and Item-based CF
- [ ] **Matrix Factorization**: SVD, SVD++, FunkSVD
- [ ] **Content-Based Filtering**: Deep feature extraction
- [ ] **Hybrid Models**: Combining collaborative + content-based

### Phase 2: Deep Learning
- [ ] **Neural Collaborative Filtering**: Deep learning for implicit feedback
- [ ] **Autoencoders**: Variational autoencoders for recommendations
- [ ] **Graph Neural Networks**: User-item interaction graphs
- [ ] **Transformer Models**: Attention-based sequential recommendations

### Phase 3: Production System
- [ ] **Real-time API**: Flask/FastAPI recommendation endpoint
- [ ] **Scalability**: Spark/Dask for large-scale processing
- [ ] **A/B Testing Framework**: Online evaluation
- [ ] **User Interface**: Web app with React frontend

### Phase 4: Research Extensions
- [ ] **Fairness Metrics**: Bias detection and mitigation
- [ ] **Serendipity**: Surprising yet relevant recommendations
- [ ] **Context-Aware**: Time, location, device factors
- [ ] **Cross-Domain**: Transfer learning across datasets

---

## 📁 Project Structure

```
reelsense-braindead/
│
├── 📓 notebooks/
│   └── ReelSense.ipynb              # Main analysis notebook
│
├── 📊 data/
│   ├── ratings.csv                  # User ratings
│   ├── movies.csv                   # Movie metadata
│   ├── tags.csv                     # User tags
│   └── links.csv                    # External IDs
│
├── 🐍 src/
│   ├── __init__.py
│   ├── preprocessing.py             # Data cleaning & feature engineering
│   ├── recommender.py               # Recommendation models
│   ├── evaluation.py                # Metrics calculation
│   ├── explainability.py            # Explanation generation
│   └── visualization.py             # EDA plots
│
├── 📁 outputs/
│   ├── ReelSense_Report.pdf         # Project report
│   ├── ReelSense_Report.docx        # Editable report
│   └── figures/                     # Generated plots
│
├── 🔧 scripts/
│   ├── download_data.py             # Dataset downloader
│   └── run_experiments.py           # Automated experiments
│
├── 📄 README.md                     # This file
├── 📋 requirements.txt              # Python dependencies
├── ⚖️ LICENSE                       # MIT License
└── 🎓 CITATION.md                   # How to cite this work
```

---

## 🛠️ Technical Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook, Git |
| **Documentation** | Markdown, LaTeX (reports) |

### Key Libraries & Versions

```python
pandas==1.5.3           # Data manipulation
numpy==1.24.2           # Numerical computing
scikit-learn==1.2.2     # ML algorithms & metrics
matplotlib==3.7.1       # Plotting
seaborn==0.12.2         # Statistical visualization
jupyter==1.0.0          # Interactive development
```

---

## 👥 Team & Acknowledgments

### Competition Details

**Event**: BrainDead - Data Analysis & ML Challenge  
**Organizer**: Department of CST, IIEST Shibpur  
**Fest**: Revelation 2K26  

### Acknowledgments

- **MovieLens Team** (GroupLens Research): For the excellent dataset
- **Revelation 2K26 Organizers**: For hosting this competition
- **IIEST Shibpur CST Department**: For promoting ML innovation

### Special Thanks

- F. Maxwell Harper & Joseph A. Konstan for MovieLens research
- Open-source community for amazing tools (Pandas, Scikit-learn)
- BrainDead competition organizers for this opportunity

---

## 📚 References

### Academic Papers

1. **Dataset Source**:  
   Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*.  
   ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19.  
   DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

2. **Recommender Systems**:  
   Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*.  
   Springer. ISBN: 978-1-4899-7637-6

3. **Explainable AI**:  
   Arrieta, A. B., et al. (2020). *Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges*.  
   Information Fusion, 58, 82-115.

### Online Resources

- **Pandas Documentation**: https://pandas.pydata.org/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **MovieLens Dataset**: https://grouplens.org/datasets/movielens/
- **Recommender Systems Course**: Coursera - University of Minnesota

---

## 📧 Contact & Links

**Repository**: [GitHub - ReelSense](https://github.com/yourusername/reelsense-braindead)  
**Documentation**: [Project Wiki](https://github.com/yourusername/reelsense-braindead/wiki)  
**Issues**: [Bug Reports & Feature Requests](https://github.com/yourusername/reelsense-braindead/issues)

---

## ⚖️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 ReelSense Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

---

<div align="center">

### 🎉 Thank You for Exploring ReelSense!

**Made with ❤️ for BrainDead Competition @ Revelation 2K26**

*Department of Computer Science and Technology*  
*IIEST Shibpur*

---

**#BrainDead26 #Revelation26 #DataScience #MachineLearning #ExplainableAI**

⭐ Star this repository if you found it helpful!

</div>
