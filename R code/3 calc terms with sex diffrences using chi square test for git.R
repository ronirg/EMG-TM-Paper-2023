library(topicmodels)
library(tm)
library(dplyr)
library(ggplot2)
Sys.setlocale("LC_ALL","en_US.utf8")

data.sample <- read.csv("visit data generated for git not real.csv") 
data.abnormal.sample <- subset(data.sample, Normal_study == FALSE)
data.abnormal.sample <- subset(data.abnormal.sample, ConclusionDelNo != '')

length(unique(data.abnormal.sample$AID))
## create a data frame that contains only the last visit per patient (used for statistical analysis)
data.abnormal.dis <- data.abnormal.sample %>% arrange(AID, desc(Visit_ID)) %>% distinct(AID, .keep_all = TRUE) 

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

##for statistical test - distinct patients - data.abnormal.dis
Conclusion_abnormal_F_dist<-clean.vec(data.abnormal.dis[data.abnormal.dis$Gender == "F",]$ConclusionDelNo)#2225
Conclusion_abnormal_M_dist<-clean.vec(data.abnormal.dis[data.abnormal.dis$Gender == "M",]$ConclusionDelNo)#3049

first.vec <- Conclusion_abnormal_F_dist
second.vec <- Conclusion_abnormal_M_dist

l1 <- length(first.vec) 
l2 <- length(second.vec)
first.vec <- paste(first.vec, collapse=" ") #paste = Concatenate vectors after converting to character. 
second.vec <- paste(second.vec, collapse=" ")
all <- c(first.vec, second.vec)
corpus <- VCorpus(VectorSource(all))
tdm <- TermDocumentMatrix(corpus)
tdm.m <- as.matrix(tdm) #convert to a matrix
colnames(tdm.m) = c("Female", "Male")
# keep words that exists in males and females more than 5 times
# a separate analysis can be performed to words that appear only in males or only in females
# common.words <- subset(tdm.m, tdm.m[,1] > 0 & tdm.m[,2] > 0)
common.words <- subset(tdm.m, tdm.m[,1] > 5 & tdm.m[,2] > 5)

# Define a function that performs the Chi-squared test and returns the result
perform.chi.square <- function(count_F, count_M, size_F, size_M) {
  # Create the contingency table
  contingency.table <- matrix(c(count_F, count_M, size_F - count_F, size_M - count_M),
                              nrow = 2, byrow = TRUE,
                              dimnames = list(c("Present", "Absent"),
                                              c("Group F", "Group M")))
  # Perform the Chi-squared test
  result <- chisq.test(contingency.table)

  # Return the p.value
  return(result$p.value)
}


#checked with distinct patients
data <- common.words[,c("Female","Male")]
data <- as.data.frame.matrix(data)

cnt_f <- nrow(subset(data.abnormal.dis, data.abnormal.dis$Gender == "F"))
cnt_m <- nrow(subset(data.abnormal.dis, data.abnormal.dis$Gender == "M"))
data$size_F <- cnt_f
data$size_M <- cnt_m

data$p_value <- apply(data, 1, function(row) {
  perform.chi.square(row["Female"], row["Male"], row["size_F"], row["size_M"])
})

data$p_value_f <- ifelse(data$p_value < 0.001, "<0.001", round(data$p_value,3))

dataSig <- data[data$p_value<0.05,]

# clac rate
dataSig$FemaleRate <- dataSig$Female/dataSig$size_F
dataSig$MaleRate <- dataSig$Male/dataSig$size_M

dataSig$propM <- ifelse(dataSig$MaleRate > dataSig$FemaleRate, TRUE, FALSE)
sorted_data_F <- dataSig[dataSig$propM == "FALSE",]
sorted_data_M <- dataSig[dataSig$propM == "TRUE",] # higher rate for males

# Sort the dataframe
sorted_data_F <- sorted_data_F[order(sorted_data_F$FemaleRate, decreasing = TRUE), ]
sorted_data_M <- sorted_data_M[order(sorted_data_M$MaleRate, decreasing = TRUE), ]

# Appendix C
write.csv(sorted_data_F,"significant diff words gender F dist gt5.csv")
write.csv(sorted_data_M,"significant diff words gender M dist gt5.csv")


