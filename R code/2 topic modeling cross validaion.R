library(topicmodels)
library(tm)
library(doParallel)
library(ggplot2)
library(scales)

# after you created dtm.conclusion.bigrams in the last script you can run this script
# remove sparse terms and empty rows from the dtm  
dtm <- removeSparseTerms(dtm.conclusion.bigrams, 0.9998) #bigrams
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
empty_docs <- Docs(dtm[rowTotals==0,]) #empty docs
data.abnormal.not.empty <- data.abnormal[-as.numeric(as.character(empty_docs)),] #original text without empty docs
dtm   <- dtm[rowTotals > 0, ]           #remove all docs without words


#---------------5-fold cross-validation---------------------

## first we perform 5-fold cross-validation for one candidate (25 topics)
burnin = 1000
iter = 1000
keep = 50
n <- nrow(dtm)
folds <- 5
k <- 25 
set.seed(1)
splitfolds <- sample(1:folds, n, replace = TRUE) # each record is assigned to fold

cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})
clusterExport(cluster, c("dtm", "k", "burnin", "iter", "keep", "splitfolds"))

results <- foreach(i = 1:folds) %dopar% {
  train_set <- dtm[splitfolds != i , ]
  valid_set <- dtm[splitfolds == i, ]
  
  fitted <- LDA(train_set, k = k, method = "Gibbs",
                control = list(seed=10, alpha=0.1, burnin = burnin, iter = iter, keep = keep) )
  return(perplexity(fitted, newdata = valid_set))
}
stopCluster(cluster)
results
mean(unlist(results)) #285.05
####################

#----------------5-fold cross-validation for different number of topics----------------
cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})


folds <- 5
set.seed(1)
splitfolds <- sample(1:folds, n, replace = TRUE)
# broad range
candidate_k <- c(2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100) # candidates for how many topics
clusterExport(cluster, c("dtm", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))

# we parallelize by the different number of topics. A processor is allocated a value
# of k, and does the cross-validation serially.  This is because it is assumed there
# are more candidate values of k than there are cross-validation folds, hence it
# will be more efficient to parallelise
system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- dtm[splitfolds != i , ]
      valid_set <- dtm[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(seed=10, alpha=0.1, burnin = burnin, iter = iter, keep = keep) )
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    return(results_1k)
  }
})
stopCluster(cluster)


results_df <- as.data.frame(results)

## Appendix A
ggplot(results_df, aes(x = k, y = perplexity)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("5-fold cross-validation of topic modeling") +
  labs(x = "Candidate number of topics", y = "Perplexity when fitting the trained model to the hold-out set")+
  scale_x_continuous(breaks = round(seq(0, 100, by = 5)))


