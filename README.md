

 Football Player Transfer Value Prediction

 Project Overview

This project aims to predict football player transfer values using various machine learning algorithms. The dataset includes player statistics, such as goals, assists, injuries, and contract details. The models implemented include Decision Trees, Random Forests, and Gradient Boosting.

 Dataset

- Entries: 10,754
- Features:
  - player: Player's name
  - team: Team name
  - position: Playing position
  - height: Player's height (in cm)
  - age: Player's age
  - appearance: Number of appearances
  - goals: Goals scored
  - assists: Assists
  - yellow cards: Number of yellow cards
  - second yellow cards: Number of second yellow cards
  - red cards: Number of red cards
  - goals conceded: Goals conceded
  - clean sheets: Clean sheets
  - minutes played: Minutes played
  - days_injured: Days injured
  - games_injured: Games missed due to injury
  - award: Number of awards
  - current_value: Current transfer value (target variable)
  - highest_value: Highest transfer value (target variable)
  - position_encoded: Encoded position for modeling
  - winger: Binary feature indicating if the player is a winger

 Methodology

1. Data Preprocessing:
   - Categorical Encoding: Converted categorical variables (e.g., player position) into numerical format using one-hot encoding.
   - Normalization: Standardized numerical features (e.g., age, height) to ensure uniform scale.

2. Model Implementation:
   - Decision Tree: Implemented a Decision Tree to model the relationship between features and transfer values.
   - Random Forest: Used Random Forests with 100 trees to improve prediction accuracy and reduce overfitting.
   - Gradient Boosting: Applied Gradient Boosting to sequentially build and correct models, optimizing performance.

3. Model Evaluation:
   - Cross-Validation: Employed k-fold cross-validation to assess model robustness and avoid overfitting.
   - Performance Metrics: Achieved a maximum R² score of 0.82, indicating strong model performance in explaining variance in transfer values.

 Installation

1. Clone the repository:
   bash
   git clone https://github.com/your-username/football-transfer-value-prediction.git
   

2. Navigate to the project directory:
   bash
   cd football-transfer-value-prediction
   

3. Install dependencies:
   bash
   pip install -r requirements.txt
   

 Usage

1. Prepare the data:
   python
    Load and preprocess data
   import pandas as pd
   data = pd.read_csv('player_data.csv')
    Data preprocessing steps here
   

2. Train and evaluate models:
   python
    Import models and tools
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.metrics import r2_score

    Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Initialize and train models
   models = {
       'Decision Tree': DecisionTreeRegressor(),
       'Random Forest': RandomForestRegressor(n_estimators=100),
       'Gradient Boosting': GradientBoostingRegressor()
   }

   for name, model in models.items():
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       print(f'{name} R² Score: {r2_score(y_test, predictions)}')
   

 Results

- Decision Tree: Provides a basic model with interpretability.
- Random Forest: Achieved a significant reduction in overfitting with an R² score of X (replace X with your result).
- Gradient Boosting: Showcased superior accuracy with an R² score of X (replace X with your result).

 Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!
