# Tpic modeling with BERTopic
# installation instructions and documentation:
# https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html
# The data file is not real and was generated manually
import nltk as nltk
import pandas as pd
import utils
import nltk
import re
from datetime import datetime
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
current_datetime = datetime.now()
# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df = pd.read_csv("visit data generated for git not real.csv", encoding='utf-8')
    #print(df.head())

    df['ConclusionDelNo'] = df['ConclusionDelNo'].astype(str)
    pattern2 = r'[?!,.]' # remove ?!,. # not a must, try with and without
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.replace(pattern2, ' ', regex=True)# delete ?,.: and !,
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.lower() # convert to lower case
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.strip()  #  remove leading and trailing white spaces
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.replace(r'\s+', ' ', regex=True) # strip white space from the middle

    # Keep rows where 'Normal_study' is False
    filtered_df = df[df['Normal_study'] == False]
    #print(filtered_df.shape)

    # keep only rows with text
    filtered_df = filtered_df[filtered_df['ConclusionDelNo'].apply(lambda x: pd.notna(x) and x != '')]
    #print(filtered_df.head())
    #print(filtered_df.shape)

    # Keep documents that have more than three words
    filtered_df = filtered_df[filtered_df['ConclusionDelNo'].str.split().apply(lambda x: len(x) > 3)]
    #print(filtered_df.shape)

    # Add an enumeration column starting from 1
    filtered_df['doc_id'] = [i + 1 for i in range(len(filtered_df))]

    #filtered_df = filtered_df.iloc[:300] #test small
    text = filtered_df.ConclusionDelNo.tolist()
    print(len(text))

    all_intents = text
    umap_model = UMAP(n_neighbors = 3, n_components = 3, min_dist=0.05, random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size = 5, min_samples = 2)
    #vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words="english") # if we want to remove stopwords from the representation, need to add to BERTopic params
    topic_model = BERTopic(umap_model = umap_model,
                           hdbscan_model = hdbscan_model,
                           language = "english", min_topic_size = 5, n_gram_range = (1, 2))
    #topic_model = BERTopic() # if we want to run with default params
    topics, probs = topic_model.fit_transform(all_intents)

    topic_model.save("bertmodel", serialization="pickle")
    #topic_model.load("bertmodel")
    topic_model = BERTopic.load("bertmodel")
    topic_info = topic_model.get_topic_info()
    topic_freq = topic_model.get_topic_freq()

    topic_info.to_csv("topic_size_and_description.csv", encoding='utf-8', index=False)
    filtered_df['Topic'] = topics

    params = topic_model.get_params()
    print("params: ", params)
    vis = topic_model.visualize_topics()
    vis.show()

    ## if we want to reduce the number of topics and keep also the original division
    # topic_model.reduce_topics(all_intents, nr_topics=15) #25, "auto"
    # topics_reduced = topic_model.topics_
    # filtered_df['Topic_Reduced'] = topics_reduced
    # topic_info_reduced = topic_model.get_topic_info()
    # topic_info_reduced.to_csv("topic_size_and_description_reduced.csv", encoding='utf-8', index=False)

    filtered_df.to_csv("visit data with topic.csv")

    print("BERTopic ended successfully")
