import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Simulated training data
data = pd.DataFrame({
    'msg_count': [100, 250, 500, 80, 300],
    'avg_word_count': [5, 10, 8, 4, 12],
    'emoji_count': [10, 30, 50, 5, 60],
    'url_count': [2, 10, 5, 0, 7],
    'avg_response_time': [40, 20, 10, 100, 15],
    'interest_score': [70, 90, 95, 50, 98]  # Simulated scores
})

X = data.drop('interest_score', axis=1)
y = data['interest_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'interest_model.pkl')
print("Model trained and saved!")
