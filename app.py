import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

# Download stopwords
nltk.download('stopwords')

# Streamlit app starts here
st.title('SMS Spam Detection')
st.write("This app performs SMS spam detection using a Naive Bayes model.")

# File upload feature to upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Text preprocessing function
    def preprocess_text(text):
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'\W', ' ', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # Stemming
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text

    # Apply preprocessing
    df['CleanMessage'] = df['Message'].apply(preprocess_text)

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Transform the text data
    X = vectorizer.fit_transform(df['CleanMessage'])
    y = df['Category']

    # Convert 'ham'/'spam' to 0/1 using LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # Define parameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    st.write("Best Parameters:", best_params)

    # Train with best parameters
    naive_bayes = MultinomialNB(**best_params)
    naive_bayes.fit(X_train, y_train)
    y_pred_nb = naive_bayes.predict(X_test)

    # Save the model and vectorizer
    joblib.dump(naive_bayes, 'naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred_nb)
    conf_matrix = confusion_matrix(y_test, y_pred_nb)
    class_report = classification_report(y_test, y_pred_nb)

    st.write(f"Accuracy: {accuracy}")
    st.write(f"Confusion Matrix:\n{conf_matrix}")
    st.write(f"Classification Report:\n{class_report}")

    # Calculate predicted probabilities
    y_pred_proba = naive_bayes.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Naive Bayes')
    plt.legend()
    st.pyplot(plt)

    # Adding the SMS input feature to predict spam or not
    sms_input = st.text_input("Enter the SMS text to check if it's spam or not:")

    if sms_input:
        # Preprocess the input
        sms_cleaned = preprocess_text(sms_input)

        # Vectorize the input
        sms_vectorized = vectorizer.transform([sms_cleaned])

        # Make the prediction
        prediction = naive_bayes.predict(sms_vectorized)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        # Show the result
        if prediction_label == 'spam':
            st.write("The SMS is **SPAM**!")
        else:
            st.write("The SMS is **NOT SPAM**.")
