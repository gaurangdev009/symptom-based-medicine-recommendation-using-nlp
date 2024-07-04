import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import joblib

# Load the training data
data = pd.read_csv("data/drugsComTrain.csv")

# Drop rows with missing values in 'condition' column
data.dropna(subset=['condition'], inplace=True)

# Assuming 'condition' is the target variable and 'review' is the text data
X = data['review']
y = data['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)

# Initialize and train the model
model = SVC(kernel='linear')
model.fit(X_train_vect, y_train)
temp=model.predict(X_test,y_test)
print(temp)
# Save the trained model to disk
joblib.dump(model, 'model/passmodel1.pkl')
