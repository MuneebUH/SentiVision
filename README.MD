# ProductPulse

ProductPulse is a Streamlit-based web application designed for sentiment analysis of Amazon product reviews. Users can upload review data, preprocess it, and use machine learning models to predict the sentiment. The app also offers various visualizations for the data and model predictions.

## Features

- **File Upload**: Supports CSV and Excel file formats for uploading review data.
- **Data Overview**: Displays a scrollable data table showing the uploaded review data.
- **Data Preprocessing**: Cleans and preprocesses the review text by removing non-alphabetical characters, converting text to lowercase, and removing stopwords.
- **Sentiment Prediction**: Uses machine learning models to predict the sentiment of the reviews (positive, negative, or neutral).
- **Visualizations**:
  - **Rating Count Bar Chart**: Displays the count of each rating (1 to 5 stars) in a bar chart.
  - **Star Rating Distribution Pie Chart**: Shows the distribution of star ratings in a pie chart.
  - **Word Cloud**: Generates a word cloud for the most frequent words in the reviews.
  - **Sentiment Count Bar Chart**: Displays the count of predicted sentiments in a bar chart.
  - **Sentiment Percentage Pie Chart**: Shows the percentage distribution of predicted sentiments in a pie chart.
  - **Confusion Matrix**: If actual sentiment labels are available, a confusion matrix is displayed to evaluate the model's performance.

## Model Performance

The project utilizes several machine learning models for sentiment analysis. Below are the accuracy scores for each model:

| Model        | Training Accuracy | Testing Accuracy |
|--------------|-------------------|------------------|
| Random Forest| 0.9965            | 0.8911           |
| Decision Tree| 0.9965            | 0.8380           |
| XGBoost      | 0.9365            | 0.8928           |

## Data Format

The application expects the uploaded file to contain the following columns:

- **star_rating**: Numeric ratings (e.g., 1 to 5).
- **review_body**: The text of the review.
