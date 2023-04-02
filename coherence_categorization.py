""" IMPORTS """
# PRE-PROCESSING
import mysql.connector, nltk, os
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

if __name__ == '__main__':
    """ PREPROCESSING """
    # Connect to the MySQL database
    mydb = mysql.connector.connect(
    host="*****", port=0000, user="*****", password="*****", database="*****")

    # Define a SQL query to retrieve the relevant fields
    query = "SELECT id, post_text, category_id, category_name, subcategory_id, subcategory_name FROM questions_text"

    # Execute the query using the database connection
    df = pd.read_sql(query, mydb)

    # Preprocess the question_text column
    stop_words = set(stopwords.words('english'))
    df['post_text'] = df['post_text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

    dictionary = Dictionary(documents=df['post_text'].str.split())

    # Create a bag of words corpus
    corpus = [dictionary.doc2bow(doc) for doc in df['post_text'].str.split()]

    # Train an LDA model on the corpus
    num_topics = len(df['category_name'].unique())
    print("Processing LDA model...\n")
    lda_model = LdaModel(corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        passes=10,
                        alpha='auto',
                        random_state=42)
    print(lda_model.print_topics())
    
    print("Processing Coherence Scores...\n")
    # Calculate the coherence score for each topic
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['post_text'].str.split(), dictionary=dictionary, coherence='c_v')
    coherence_scores = coherence_model_lda.get_coherence_per_topic()

    # Print the topics and their coherence scores
    print(coherence_scores)




