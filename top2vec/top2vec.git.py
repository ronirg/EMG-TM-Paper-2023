import nltk as nltk
import pandas as pd
import utils
import nltk
from sentence_transformers import SentenceTransformer
import re
from top2vec import Top2Vec

# Function to remove a specific string from the end of each sentence

if __name__ == '__main__':
    min_words = 3
    # load the data
    df = pd.read_csv("visit data generated for git not real.csv", encoding='utf-8')
    # if we train the model with doc2vec it is recommended to train at least 1000 records
    #print(df.head())

    df['ConclusionDelNo'] = df['ConclusionDelNo'].astype(str)

    pattern1 = r'[?!,.]'
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.replace(pattern1, ' ', regex=True)# delete ?,.: and !, # not a must
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.lower() # convert to lower case
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.strip()  # remove leading and trailing white spaces
    df['ConclusionDelNo'] = df['ConclusionDelNo'].str.replace(r'\s+', ' ', regex=True) # strip white space from the middle
    # Didn't remove stopwords and didn't perform stemming as the documentation suggest
    # Keep rows where 'Normal_study' is False
    filtered_df = df[df['Normal_study'] == False]
    #print(filtered_df.shape)

    # keep only rows with text in 'ConclusionDelNo'
    filtered_df = filtered_df[filtered_df['ConclusionDelNo'].apply(lambda x: pd.notna(x) and x != '')]
    #print(filtered_df.shape)

    # Keep documents that have more than three words
    filtered_df = filtered_df[filtered_df['ConclusionDelNo'].str.split().apply(lambda x: len(x) > min_words)] #3
    #print(filtered_df.shape)

    # Add an enumeration column starting from 1
    filtered_df['doc_id'] = [i + 1 for i in range(len(filtered_df))]

    #filtered_df = filtered_df.iloc[:1000] #test on a smaller dataset
    text = filtered_df.ConclusionDelNo.tolist()
    print(len(text))

## added random_state for repruducable results
# the other parameters were kept as default
# the results will be the same only when using pre-trained embbedings, doc2vec will still produce slightly diferent results each time
#     umap_args = {'n_neighbors': 15,
#                  'n_components': 5,
#                  'metric': 'cosine',
#                  'random_state': 42} # default for paper

    umap_args_sample = {'n_neighbors': 3,
                 'n_components': 3,
                 'min_dist': 0.05,
                 'metric': 'cosine',
                 'random_state': 42} # for small sample

    # hdbscan_args = {'min_cluster_size': 15,
    #                 'min_samples': 5, # not defined in default
    #                 'metric': 'euclidean',
    #                 'cluster_selection_method': 'eom'} #deault

    hdbscan_args_sample = {'min_cluster_size': 5,
                    'min_samples': 2, # not defined in default
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'} #deault

    # In order to create the model we run the following lines, after we saved the model we can comment these lines
    # and load the model
    # we used the default doc2vec embbedings which are appropriate when you have unique terms
    # we examined also 'universal-sentence-encoder' pre-trained embeddings
    # ngram_vocab adds phrases to topic description (default is False)
    # min_count - minimum term frequency, we tried 10, 15, 20, 30, 50

    # Change "deep-learn" to "fast-learn" or "learn" if you want to test the code.It will finish fast but will be less accurate
    # params in paper
    #model = Top2Vec(documents=text, speed="deep-learn", workers=8, min_count=15, umap_args=umap_args, ngram_vocab=False)

    # params for sample data
    model = Top2Vec(documents=text, speed="deep-learn", workers=8, min_count=5, umap_args=umap_args_sample,
                    hdbscan_args=hdbscan_args_sample, ngram_vocab=False)
    # if we want to use pre-trained embeddings we need to:
    # pip install top2vec[sentence_encoders]
    # pip install top2vec[sentence_transformers]
    # and set embedding_model='universal-sentence-encoder' in Top2Vec parameters
    # model = Top2Vec(documents=all_intents, speed="deep-learn", workers=8, min_count=10, umap_args=umap_args,
    #                 embedding_model='universal-sentence-encoder')

    model.save("top2vec_model_deep_learn_small")

    model = Top2Vec.load("top2vec_model_deep_learn_small")
    num_topics = model.get_num_topics()
    if num_topics > 25:
        model.hierarchical_topic_reduction(num_topics = 25)
    else:
        model.hierarchical_topic_reduction(num_topics=num_topics-1)

    topics_hierarchy_list = model.get_topic_hierarchy()
    topic_sizes, topic_nums = model.get_topic_sizes()
    print("num_tpics = ",num_topics)
    print("sizes = ",topic_sizes)
    topic_words, word_scores, topic_nums = model.get_topics(num_topics)
    #print(word_scores)
    #print(topic_nums)
    #print(topic_words)

    #top_words_per_topic = model.get_topics(10)
    #################
    # Create a dataframe to store document-topic mapping
    document_topic_mapping = []
    for doc_id, topic_id in enumerate(model.doc_top): #add one - same as in filtered df
        document_topic_mapping.append({"doc_id": doc_id+1, "Document": model.documents[doc_id], "Topic": topic_id})

    document_topic_df = pd.DataFrame(document_topic_mapping)

    # Save document-topic mapping to a CSV file
    output_csv = "document_topic_mapping.csv"
    document_topic_df.to_csv(output_csv, encoding='utf-8', index=False)
    print(f"Document-topic mapping saved to {output_csv}")
    # Merge dataframes using common column "DocumentID"
    merged_df = document_topic_df.merge(filtered_df, on="doc_id", how="left")

    # Save merged dataframe to a CSV file
    merged_df.to_csv("merged_document_topic_data.csv", encoding='utf-8', index=False)
    topic_words_df = pd.DataFrame(topic_words)
    topic_words_df.to_csv("topic_terms.csv", encoding='utf-8', index=False)

    pd.DataFrame(topic_sizes).to_csv("topic_sizes.csv", encoding='utf-8', index=False)
    pd.DataFrame(topics_hierarchy_list).to_csv("topics_hierarchy_list.csv", encoding='utf-8', index=False)

    ################################
    ### We reduce the number of topics to 25 (in order to compare to LDA result)
    num_topics_reduced = model.get_num_topics(reduced=True)
    topic_sizes_reduced, topic_nums_reduced = model.get_topic_sizes(reduced=True)
    topic_words_reduced, word_scores_reduced, topic_nums_reduced = model.get_topics(reduced=True, num_topics=num_topics_reduced)
    #################
    # Create a dataframe to store document-topic mapping
    document_topic_mapping_reduced = []
    for doc_id, topic_id_reduced in enumerate(model.doc_top_reduced): #add one - same as in filtered df
        document_topic_mapping_reduced.append({"doc_id": doc_id+1, "Document": model.documents[doc_id], "Topic": topic_id_reduced})

    document_topic_reduced_df = pd.DataFrame(document_topic_mapping_reduced)

    # Save document-topic mapping to a CSV file
    output_reduced_csv = "document_topic_mapping_reduced.csv"
    document_topic_reduced_df.to_csv(output_reduced_csv, encoding='utf-8', index=False)

    print(f"Document-topic mapping saved to {output_reduced_csv}")

    merged_reduced_df = document_topic_reduced_df.merge(filtered_df, on="doc_id", how="left")
    # Save merged dataframe to a CSV file
    merged_reduced_df.to_csv("merged_document_topic_reduced_data.csv", encoding='utf-8', index=False)
    topic_words_reduced_df = pd.DataFrame(topic_words_reduced)
    topic_words_reduced_df.to_csv("topic_terms_reduced.csv", encoding='utf-8', index=False)
    pd.DataFrame(topic_sizes_reduced).to_csv("topic_sizes_reduced.csv",encoding='utf-8', index=False)

    print("top2vec ended successfully.")
