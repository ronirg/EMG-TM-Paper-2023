library(topicmodels)
library(tm)
library(textstem)
library(stringr)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(RWeka)
Sys.setlocale("LC_ALL","en_US.utf8")

###################################
## upload data file
data <- read.csv("visit data generated for git not real.csv") 
## keep only abnormal visits
data.abnormal.sample <- subset(data, Normal_study == FALSE)

## keep rows with Conclusion text after deleting negation phrases like: "There is no evidence..."
data.abnormal.sample <- subset(data.abnormal.sample, ConclusionDelNo != '')

length(unique(data.abnormal.sample$AID))
## create a data frame that contains only the last visit per patient (used for statistical analysis)
data.abnormal.dis.sample <- data.abnormal.sample %>% arrange(AID, desc(Visit_ID)) %>% distinct(AID, .keep_all = TRUE) 

######################

custom.stopwords <- c(stopwords("english"), 'this', 'study', 'shows', 'electrodiagnostic', 'evidence', 'conclusion', 'based', 'finding')

# create a custom lexicon for the lemmatization, change the lemma of "left"
# library(lexicon)
# newlexicon <- lexicon::hash_lemmas
# newlexicon["left"]$lemma <- "left" ###change the lemma of "left" to "left"

# function to perform generic cleaning
clean.data <- function(text){
  myCorpus <- VCorpus(VectorSource(text))
  myCorpus <- tm_map(myCorpus, content_transformer(tolower))# convert to lower case
  myCorpus <- tm_map(myCorpus, removeWords, custom.stopwords) #remove stopwords
  #myCorpus <- tm_map(myCorpus, removeNumbers)# remove numbers
  myCorpus <- tm_map(myCorpus, removePunctuation, ucp = TRUE)# remove punctuation  #ucp a logical specifying whether to use Unicode character properties for determining punctuation characters. If FALSE (default), characters in the ASCII [:punct:] class are taken; if TRUE, the characters with Unicode general category P (Punctuation).
  #myCorpus <- tm_map(myCorpus, removePunctuation)
  myCorpus <- tm_map(myCorpus, stripWhitespace)# remove extra whitespace
  myCorpus <- tm_map(myCorpus, content_transformer(lemmatize_strings)) # lemmatize strings (changes left to leave)
  #myCorpus <- tm_map(myCorpus, content_transformer(function(x)lemmatize_strings(x,dictionary = newlexicon))) ## use an adjusted lexicon (doesn't change left to leave)
  myCorpus <- tm_map(myCorpus, removeWords, custom.stopwords) #remove custom stopwords again after lemmatization
  return(myCorpus)
}
conclusion.corpus <- clean.data(data.abnormal.sample$ConclusionDelNo)

for(i in 1:20){
  print(conclusion.corpus[[i]][1])
}

## create DTM with Bi-grams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
dtm.conclusion.bigrams <- DocumentTermMatrix(conclusion.corpus, control = list(tokenize = BigramTokenizer))
inspect(dtm.conclusion.bigrams)#19280

########################################################
## evaluate the number of topics using the ldatuning library
library(ldatuning)
find.topics <- function(dtm, s){
  dtm <- removeSparseTerms(dtm, s)
  rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
  dtm   <- dtm[rowTotals> 0, ]           #remove all docs without words
  set.seed(10)
  result <- FindTopicsNumber(
    dtm,
    topics = seq(from = 2, to = 30, by = 1),
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
    method = "Gibbs",
    control = list(seed = 10,alpha=0.1),
    mc.cores = 2L,
    verbose = TRUE
  )
  return(result)
}
res <- find.topics(dtm.conclusion.bigrams, 0.9998)
# Appendix B - ldatuning plot
FindTopicsNumber_plot(res)

## remove empty rows from the dtm and apply Gibbs LDA with 25 topics 
dtm <- removeSparseTerms(dtm.conclusion.bigrams, 0.9998) #bigrams
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
empty_docs <- Docs(dtm[rowTotals==0,]) #empty docs
data.abnormal.sample.not.empty <- data.abnormal.sample[-as.numeric(as.character(empty_docs)),] #original text without empty docs
dtm   <- dtm[rowTotals > 0, ]           #remove all docs without words
#set.seed(10)
lda.bigrams <- LDA(dtm, k = 25, control=list(seed=10,alpha=0.1),method = "Gibbs") #lower alpha, we assume documents contain fewer topics

topterms <- terms(lda.bigrams, 10) # top 10 terms for each topic
# save in csv to create Figure 6 and Appendix D
write.csv(topterms,"topterms 25 bigrams sample.csv")

max.probs.doc <- apply(posterior(lda.bigrams)$topics, 1, max) # maximum probability
max.topic.doc <- colnames(posterior(lda.bigrams)$topics)[apply(posterior(lda.bigrams)$topics, 1, which.max)] # topic with maximum probability
# Create a data frame with the topic of each document and the topic probability
max.topic.per.doc <- data.frame(Document = 1:nrow(data.abnormal.sample.not.empty), AID = data.abnormal.sample.not.empty$AID, Visit_ID = data.abnormal.sample.not.empty$Visit_ID, MaxValue = max.probs.doc, MaxTpoic = max.topic.doc,
                                ConclusionDelNo = data.abnormal.sample.not.empty$ConclusionDelNo)
write.csv(max.topic.per.doc, "LDA docs assignment with max prob sample.csv")

num_topics = lda.bigrams@k
posterior_terms <- posterior(lda.bigrams)$terms

# Get the top 10 terms and their probabilities for each topic
top_terms <- lapply(1:num_topics, function(topic_index) {
  topic_terms <- posterior_terms[topic_index,]
  top_terms_indices <- order(topic_terms, decreasing = TRUE)[1:10]
  top_terms_probs <- topic_terms[top_terms_indices]
  data.frame(
    Topic = paste0("Topic", topic_index),
    Term = colnames(posterior_terms)[top_terms_indices],
    Probability = top_terms_probs
  )
})

# Combine the list of data frames into a single data frame
top.terms.df.probs <- do.call(rbind, top_terms)
write.csv(top.terms.df.probs, "LDA top terms per topic with probs sample.csv")

# topic assignments
topics <- topics(lda.bigrams)

topterms.df <- as.data.frame(topterms)
size <- as.data.frame(table(topics))
topic.label <- c('CTS','CTS-Mild','Multiple Entrapment Neuropathies', 
                 'Active Lumbosacral Radiculopathy', 'DP with axonal loss', 'LDPN with Active EMG Changes', 'Brachial Plexopathy',
                 'LDPN Symmetric Sensory','Peroneal Neuropathy','DP with axonal loss severe',
                 'DP with axonal loss','ALS','Active Radiculopathy','Superficial Peroneal and Mixed Neuropathies',
                 'Chronic Radiculopathy Cervical','Ulnar Neuropathy Elbow',
                 'Test Comparison','Myopathy','LDPN Sensory Polyneuropathy',
                 'Myasthenia/NMJ','Sensory Neuropathy','Chronic Radiculopathy Lumbar',
                 'Median Neuropathy Wrist','CTS Moderate','Chronic L5 Radiculopathy')
topterms.df.size.label <- rbind(size$Freq, topic.label, topterms.df)
topterms.df.size.label.t <- t(topterms.df.size.label)

write.csv(topterms.df.size.label.t,"topterms 25 bigrams new with size and label sample.csv")
write.csv(topterms.df.size.label,"topterms 25 bigrams new with size and label plane sample.csv")


### Figure 3
age.gender <- data.abnormal.sample %>%
  ggplot( aes(x=Age, fill=Gender)) +
  geom_histogram(breaks = seq(0, 100, by = 2), color="#e9ecef", alpha=0.8, position = 'identity') +
  scale_fill_manual(values=c("#333333", "#cccccc")) + theme_bw() +
  scale_x_continuous(breaks = seq(0, 100, 10)) + labs(y = "Number of visits") + 
  theme(legend.position="bottom")
age.gender

#Fig 4
dtm4 <- removeSparseTerms(dtm.conclusion.bigrams, 0.9998)
rowTotals <- apply(dtm4 , 1, sum) #Find the sum of words in each Document
dtm4   <- dtm4[rowTotals> 0, ]           #remove all docs without words
m4 <- as.matrix(dtm4)
tdm4.new <- as.TermDocumentMatrix(t(m4), weighting = weightTf)
m4 <- as.matrix(tdm4.new)
v4 <- sort(rowSums(m4),decreasing=TRUE)
d4 <- data.frame(word = names(v4),freq=v4)
head(d4, 10)
write.csv(d4,"Fig4 freq bigrams.csv")

# Figure 4
bar.plot.bigrams <- ggplot(d4[1:10,], aes(x = freq, y = reorder(word, freq))) +
  geom_bar(stat = "identity", fill = "gray") +
  labs(x = "Frequency", y = "Bigram") +
  theme_minimal()

#####################################################create tdm.m for pyramid
clean.vec<-function(text.vec){
  text.vec <- tolower(text.vec)
  text.vec <- removeWords(text.vec, custom.stopwords)
  text.vec <- removePunctuation(text.vec)
  #text.vec <- removeNumbers(text.vec)
  #text.vec <- lemmatize_strings(text.vec)
  text.vec <- removeWords(text.vec, custom.stopwords)
  text.vec <- stripWhitespace(text.vec)
  return(text.vec)
}
custom.stopwords <- c(stopwords("english"),'study', 'shows', 'electrodiagnostic', 'evidence', 'conclusion', 'based', 'finding')

Conclusion_abnormal_F<-clean.vec(data.abnormal.sample[data.abnormal.sample$Gender == "F",]$ConclusionDelNo)#2584
Conclusion_abnormal_M<-clean.vec(data.abnormal.sample[data.abnormal.sample$Gender == "M",]$ConclusionDelNo)#3512

first.vec <- Conclusion_abnormal_F
second.vec <- Conclusion_abnormal_M

l1 <- length(first.vec) #2584 female records
l2 <- length(second.vec)#3512 male records
first.vec <- paste(first.vec, collapse=" ") #paste = Concatenate vectors after converting to character. 
second.vec <- paste(second.vec, collapse=" ")
all <- c(first.vec, second.vec)
length(all)
corpus <- VCorpus(VectorSource(all))
tdm <- TermDocumentMatrix(corpus)
tdm.m <- as.matrix(tdm) #convert to a matrix
colnames(tdm.m) = c("Female", "Male")

#################################################################
options(scipen = 999)
library(plotrix)
#only terms that appear in both documents
common.words <- subset(tdm.m, tdm.m[,1] > 0 & tdm.m[,2] > 0) #578 terms (out of 4049)
common.words.f.only <- subset(tdm.m, tdm.m[,1] > 0 & tdm.m[,2] == 0)
common.words.m.only <- subset(tdm.m, tdm.m[,1] == 0 & tdm.m[,2] > 0)
tail(common.words)

males.norm <- common.words[,2]*l1/l2
diff.norm <- males.norm - common.words[,1]
diff.norm.abs <- abs(diff.norm)
difference <- abs(common.words[,1] - common.words[,2]) #find the difference between the frequencies
sumRows <- rowSums(common.words)
diff_rate <- (difference/sumRows)*100
# add a new column that contains the difference
common.words <- cbind(common.words, difference, sumRows, diff_rate, males.norm, diff.norm, diff.norm.abs)
#write.csv(common.words,"common.words.csv")
common.words <- as.data.frame(common.words[order(common.words[,8],decreasing = TRUE), ])
common.words.m <- common.words[common.words$diff.norm > 0,]
common.words.f <- common.words[common.words$diff.norm < 0,]
head(common.words)
#select the first 25 term values
top25.df <- data.frame(x = common.words[1:25, 1],y = common.words[1:25, 2],
                       labels = rownames(common.words[1:25, ]))

top25.df.m <- data.frame(x = common.words.m[1:25, 1],y = round(common.words.m[1:25, "males.norm"]),
                       labels = rownames(common.words.m[1:25, ]))
top25.df.f <- data.frame(x = common.words.f[1:25, 1],y = round(common.words.f[1:25, "males.norm"]),
                       labels = rownames(common.words.f[1:25, ]))
dev.new()
#create the pyramid plot
# x contains the female frequency, y the male frequency
###Figure 5 for paper (gray)
#F
pyramid.plot(top25.df.f$x, top25.df.f$y, labels = top25.df.f$labels,
             gap = 200, top.labels = c("Female", "Words", "Male Norm"), 
             lxcol=hsv(0, 0, seq(0.4,0.8,length.out=length(top25.df.f$labels))),
             rxcol=hsv(0, 0, seq(0.4,0.8,length.out=length(top25.df.f$labels))),
             show.values = T, space=0.3, ndig=0, xlim=c(1500,1500),
             main = "Words in Common, more popular among females", unit = NULL) 
#M
pyramid.plot(top25.df.m$x, top25.df.m$y, labels = top25.df.m$labels,
             gap = 300, top.labels = c("Female", "Words", "Male Norm"),
             lxcol=hsv(0, 0, seq(0.4,0.8,length.out=length(top25.df.f$labels))),
             rxcol=hsv(0, 0, seq(0.4,0.8,length.out=length(top25.df.f$labels))),
             show.values = T, space=0.3, ndig=0, xlim=c(2200,2200),
             main = "Words in Common, more popular among males", unit = NULL) 

###################################################################################################################

