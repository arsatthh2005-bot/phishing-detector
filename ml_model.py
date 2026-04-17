from sklearn.linear_model import LogisticRegression

# Training data (simple demo)
X = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 0],
    [0, 1, 0]
]

y = [1, 1, 0, 0]

model = LogisticRegression()
model.fit(X, y)

def predict_ml(features):
    return model.predict([features])[0]