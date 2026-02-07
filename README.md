# ReelSense: Explainable Movie Recommender System

> **BrainDead Competition | Revelation 2K26**  
> Department of Computer Science and Technology, IIEST Shibpur

## 📌 Overview

ReelSense is an explainable movie recommendation system built with the MovieLens dataset. Unlike traditional black-box recommenders, it provides clear natural language explanations for every recommendation.

**Key Feature**: *"We recommend 'Fight Club (1999)' because you liked Godfather, The (1972) and both are 'Drama, Thriller' films."*

## 🎯 Features

- ✅ **Explainable Recommendations** - Natural language explanations linking suggestions to user preferences
- ✅ **Comprehensive Metrics** - Precision, Recall, NDCG, Coverage, Diversity evaluation
- ✅ **Feature Engineering** - 1,496 combined features (genres + tags) with cosine similarity
- ✅ **Time-based Split** - Leave-last-1 evaluation for realistic testing
- ✅ **EDA & Visualizations** - Complete exploratory data analysis with insights

## 📊 Dataset

**Source**: MovieLens Latest Small Dataset

| Component | Details |
|-----------|---------|
| Ratings | 100,836 ratings from 610 users |
| Movies | 9,742 movies with genres |
| Tags | 3,683 user-generated tags |
| Training Set | 100,226 ratings |
| Test Set | 610 ratings (last per user) |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone [https://github.com/yourusername/reelsense.git](https://github.com/JK110/ReelSense.git)
cd reelsense

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run notebook
jupyter notebook ReelSense.ipynb
```

### Usage Example

```python
import pandas as pd
from reelsense import ReelSenseRecommender

# Load data
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Get recommendations
recommender = ReelSenseRecommender()
recommender.fit(ratings, movies)
recommendations = recommender.recommend(user_id=1, n=10)

# Generate explanation
explanation = recommender.explain(user_id=1, movie_id=318)
print(explanation)
```

## 📈 Results

### Evaluation Metrics (K=10)

| Metric | Value | Insight |
|--------|-------|---------|
| Precision@10 | 0.0018 | Non-personalized baseline |
| Recall@10 | 0.0180 | Limited coverage |
| NDCG@10 | 0.0096 | Room for improvement |
| Catalog Coverage | 0.0010 | Only 0.1% of catalog |
| Intra-List Diversity | 0.8079 | ✅ High diversity |
| Novelty Score | 0.2069 | Popular items focus |

### Top 10 Recommended Movies

| Rank | Movie | Avg Rating | Count |
|------|-------|------------|-------|
| 1 | Shawshank Redemption, The (1994) | 4.43 | 315 |
| 2 | Godfather, The (1972) | 4.28 | 189 |
| 3 | Fight Club (1999) | 4.27 | 218 |
| 4 | Cool Hand Luke (1967) | 4.27 | 57 |
| 5 | Dr. Strangelove (1964) | 4.26 | 96 |

### Key Insights from EDA

- **Rating Distribution**: Peak at 4.0-5.0 (users rate what they enjoy)
- **Popular Genres**: Drama (25,606), Comedy (16,870), Action (13,234)
- **Highest Rated**: Film-Noir (4.12), Documentary (3.95), War (3.89)
- **User Behavior**: Long-tail distribution (few power users, many casual)
- **Movie Popularity**: 1% of movies get 25% of ratings

## 🔬 Methodology

### 1. Data Preprocessing
- Timestamp conversion to datetime
- Time-based train-test split (leave-last-1)
- Genre cleaning and one-hot encoding (21 features)
- Tag normalization and one-hot encoding (1,476 features)
- Cosine similarity matrix (9,742 × 9,742)

### 2. Baseline Model
- Popularity-based recommender
- Minimum 50 ratings threshold
- Ranked by average rating

### 3. Explainability Engine
- Identify user's top-rated movies
- Find shared genres/tags with recommendation
- Generate natural language explanation

### 4. Evaluation Framework
- Ranking metrics (Precision, Recall, NDCG, HR, MAP)
- Diversity metrics (Coverage, Intra-list diversity, Novelty)

## 🔮 Future Work

- [ ] Collaborative Filtering (User-based, Item-based)
- [ ] Matrix Factorization (SVD, FunkSVD)
- [ ] Content-Based Filtering
- [ ] Hybrid Models
- [ ] Deep Learning approaches (NCF, Autoencoders)
- [ ] Real-time API with Flask/FastAPI
- [ ] Web UI for user interaction

## 📁 Project Structure

```
reelsense/
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
├── notebooks/
│   └── ReelSense.ipynb
├── outputs/
│   ├── ReelSense_Report.pdf
│   └── ReelSense_Report.docx
├── README.md
└── requirements.txt
```

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms & metrics
- **Matplotlib & Seaborn** - Visualization
- **Jupyter** - Interactive development

## 📚 References

**Dataset**:  
Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), Article 19. https://doi.org/10.1145/2827872

**Libraries**:
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- Scikit-learn: https://scikit-learn.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

## 🏆 Competition

**Event**: BrainDead - Data Analysis & Machine Learning Challenge  
**Organizer**: Department of Computer Science and Technology, IIEST Shibpur  
**Fest**: Revelation 2K26

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

**#BrainDead26 #Revelation26 #DataScience #MachineLearning #ExplainableAI**

Made with ❤️ for Revelation 2K26 | IIEST Shibpur
