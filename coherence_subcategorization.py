""" IMPORTS """
# PRE-PROCESSING
import mysql.connector, nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

if __name__ == "__main__":
    """ PREPROCESSING """
    # Connect to the MySQL database
    mydb = mysql.connector.connect(host="*****", port=0000, user="****", password="*****", database="****")

    # Define a SQL query to retrieve the relevant fields
    query = "SELECT id, post_text, category_id, category_name, subcategory_id, subcategory_name FROM questions_text"

    # Execute the query using the database connection
    df = pd.read_sql(query, mydb)

    # Preprocess the question_text column
    stop_words = set(stopwords.words('english'))
    df['post_text'] = df['post_text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

    # Create a dictionary for each category and train an LDA model on the corpus for that category
    categories = df['category_name'].unique()
    for category in categories:
        print("Processing category:", category)
        category_df = df.loc[df['category_name'] == category]
        subcategories = category_df['subcategory_name'].unique()
        num_topics = len(subcategories)
        print("Number of topics for category:", num_topics)
        
        # Create dictionary and corpus for the category
        category_dictionary = Dictionary(documents=category_df['post_text'].str.split())
        category_corpus = [category_dictionary.doc2bow(doc) for doc in category_df['post_text'].str.split()]

        # Train an LDA model on the corpus for the category
        category_lda_model = LdaModel(corpus=category_corpus,
                                    id2word=category_dictionary,
                                    num_topics=num_topics,
                                    passes=10,
                                    alpha='auto',
                                    random_state=42)
        coherence_model_lda = CoherenceModel(model=category_lda_model, texts=category_df['post_text'].str.split(), dictionary=category_dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()

        # append to txt file created
        with open('file/to/path','a') as f:
            f.write(f"{category}: {coherence_score}\n")
