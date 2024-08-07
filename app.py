import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Function to load data from uploaded file
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                df = None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = None
    else:
        df = None
    return df


# Title
# st.title("SentiVision")

# Title with center alignment
st.markdown("<h1 style='text-align: center;'>SentiVision</h1>", unsafe_allow_html=True)

# Display an image with adjusted size
# st.image("image.jpg", caption="Example Image", width=500)  # Adjust width as needed

# Display an image with center alignment and specific size
image_path = "image.png"  #
st.image(image_path, use_column_width='auto')

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Load the data
df = load_data(uploaded_file)

if df is not None and 'star_rating' in df.columns and 'review_body' in df.columns:
    # Data Overview with scrolling enabled
    st.write("### Data Overview")
    st.dataframe(df, height=300, width=900)

    # Load model and preprocessors
    @st.cache_resource
    def load_model_and_preprocessors():
        model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))
        cv = pickle.load(open('countVectorizer.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model_xgb, cv, scaler

    model_xgb, cv, scaler = load_model_and_preprocessors()

    # Preprocess reviews
    STOPWORDS = set(stopwords.words('english'))
    
    def preprocess_reviews(df):
        corpus = []
        stemmer = PorterStemmer()
        for i in range(df.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', df.iloc[i]['review_body'])
            review = review.lower().split()
            review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
            review = ' '.join(review)
            corpus.append(review)
        return corpus

    corpus = preprocess_reviews(df)
    X = cv.transform(corpus).toarray()
    X_scl = scaler.transform(X)

    # Predict sentiments
    y_preds = model_xgb.predict(X_scl)
    df['predicted_sentiment'] = y_preds 

    # Visualization: Rating Count
    st.write("### Rating Count")
    fig, ax = plt.subplots()
    df['star_rating'].value_counts().plot.bar(color='blue', ax=ax)
    ax.set_title('Rating Count', fontsize=12)
    ax.set_xlabel('Ratings', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    st.pyplot(fig)

    # Visualization: Star Rating Distribution Pie Chart
    st.write("### Star Rating Distribution")
    rating_distribution = df['star_rating'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        rating_distribution,
        labels=rating_distribution.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},  # Set font size for labels
        pctdistance=0.85  # Distance of percentages from the center
    )

    # Adjust the size of the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(12)  # Change the font size for percentages

    ax.set_title('Star Rating Distribution', fontsize=14)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)

    # Visualization: Word Cloud for all reviews
    st.write("### Word Cloud for all reviews")
    df['review_body'] = df['review_body'].astype(str)
    reviews = " ".join([review for review in df['review_body']])
    wc = WordCloud(background_color='white', max_words=50).generate(reviews)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title('Word Cloud for all reviews', fontsize=14)
    ax.axis('off')
    st.pyplot(fig)

    st.write("### Predicted Sentiment")
    st.dataframe(df, height=300, width=900)

    # Visualization: Sentiment Count Bar Chart
    st.write("### Sentiment Count")
    fig, ax = plt.subplots()
    df['predicted_sentiment'].value_counts().plot.bar(color='orange', ax=ax)
    ax.set_title('Sentiment Count', fontsize=12)
    ax.set_xlabel('Sentiment', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    st.pyplot(fig)

    # Visualization: Sentiment Percentage Pie Chart
    st.write("### Sentiment Percentage")
    sentiment_distribution = df['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        sentiment_distribution,
        labels=sentiment_distribution.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},  # Set font size for labels
        pctdistance=0.85  # Distance of percentages from the center
    )

    # Adjust the size of the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(12)  # Change the font size for percentages

    ax.set_title('Sentiment Percentage', fontsize=14)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)



    # Confusion Matrix
    st.write("### Confusion Matrix")
    if 'sentiment' in df.columns:
        y_actual = df['sentiment']
        cm = confusion_matrix(y_actual, y_preds, labels=model_xgb.classes_)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_xgb.classes_)
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14)
        st.pyplot(fig)
    else:
        st.write("Actual sentiment labels are not available for confusion matrix. To plot Confusion Matrix upload file with actual 'sentiment' column.")

else:
    st.write("Please upload a CSV or Excel file containing 'star_rating' and 'review_body' columns.")


st.markdown("""
    <br>
    <center>
        <strong>Developed by Muneeb Ul Hassan</strong>
        <a href="https://www.linkedin.com/in/muneeb-ul-hassan-machine-learning-expert/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:20px;height:20px;margin-left:8px;">
        </a>
    </center>
    """, unsafe_allow_html=True)
