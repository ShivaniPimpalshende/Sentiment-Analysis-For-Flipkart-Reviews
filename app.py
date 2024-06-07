from flask import Flask, render_template, request, jsonify
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__, static_url_path='/static')

# Load datasets
df_Electronics = pd.read_csv('Electronics.csv')
df_new_products = pd.read_csv('new_products.csv')
sid = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(review):
    sentiment_score = sid.polarity_scores(str(review))['compound']
    if sentiment_score > 0.05:
        return 'positive'
    elif sentiment_score < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Route to get unique product names
@app.route('/get_products', methods=['POST'])
def get_products():
    csv_file = request.form['csvFile']
    if csv_file == 'Electronics':
        products = df_Electronics['Product_name'].unique().tolist()
    elif csv_file == 'new_products':
        products = df_new_products['Product_name'].unique().tolist()
    else:
        return 'Invalid CSV file selected.'
    return jsonify(products)

# Route to analyze sentiment and calculate accuracy
@app.route('/analyze', methods=['POST'])
def analyze():
    csv_file = request.form['csvFile']
    input_product = request.form['inputProduct']
    
    if csv_file == 'Electronics':
        df_selected = df_Electronics
    elif csv_file == 'new_products':
        df_selected = df_new_products
    else:
        return 'Invalid CSV file selected.'
    
    if input_product == 'Select from list':
        product_name = request.form['productName']
    else:
        product_name = input_product
    
    # Actual sentiments from the dataset
    actual_sentiments = df_selected[df_selected['Product_name'] == product_name]['Sentiment']
    
    # Predicted sentiments using sentiment analysis function
    predicted_sentiments = df_selected[df_selected['Product_name'] == product_name]['Review'].apply(analyze_sentiment)

    # Calculate accuracy
    accuracy = accuracy_score(actual_sentiments, predicted_sentiments) * 100

    # Count of sentiments
    sentiment_counts = predicted_sentiments.value_counts().to_dict()

    return jsonify({
        'accuracy': accuracy,
        'sentiment_counts': sentiment_counts
    })

# Route to render the index.html file
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
