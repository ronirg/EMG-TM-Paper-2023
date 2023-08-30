library(ggplot2)
library(dplyr)
library(data.table)
library(rstatix)
library(ggpubr)

data.topic.all <- read.csv("abnormal data with 25 topic bigrams git generated data.csv")
data.topic.all$topics.labeled <- recode(data.topic.all$Topic, '1'='1.CTS','2'='2.CTS-Mild','3'='3.Multiple Entrapment Neuropathies', 
                                    '4'='4.Active Lumbosacral Radiculopathy', '5'='5.DP with axonal loss', '6'='6.LDPN with Active EMG Changes', '7'='7.Brachial Plexopathy',
                                    '8'='8.LDPN Symmetric Sensory','9'='9.Peroneal Neuropathy','10'='10.DP with axonal loss severe',
                                    '11'='11.DP with axonal loss','12'='12.ALS','13'='13.Active Radiculopathy','14'='14.Superficial Peroneal/Mixed Neurop.',
                                    '15'='15.Chronic Radiculopathy Cervical','16'='16.Ulnar Neuropathy Elbow',
                                    '17'='17.Test Comparison','18'='18.Myopathy','19'='19.LDPN Sensory Polyneuropathy',
                                    '20'='20.Myasthenia/NMJ','21'='21.Sensory Neuropathy','22'='22.Chronic Radiculopathy Lumbar',
                                    '23'='23.Median Neuropathy Wrist','24'='24.CTS Moderate','25'='25.Chronic L5 Radiculopathy')


data.topic.all$topics.labeled <- as.factor(as.character(data.topic.all$topics.labeled))
data.topic.all$topics.labeled <- fct_reorder(data.topic.all$topics.labeled, data.topic.all$Topic) ##change factor order by another variable (Topic)
levels(data.topic.all$topics.labeled)

##############keep one record per patient
data.topic <- data.topic.all %>% arrange(AID, desc(Visit_ID)) %>% distinct(AID, .keep_all = TRUE) ###distinct patients for statistical analysis

Wilcoxon.age.all <- compare_means(Age ~ Gender, data = data.topic)##no significant difference in age between males and females
Wilcoxon.age.all

###compare significant age differences by gender for each topic
table(data.topic$Topic, data.topic$Gender)
Wilcoxon.age.gender <- compare_means(Age ~ Gender, data = data.topic, group.by = "Topic")
effect_size_age <- data.topic %>% group_by(Topic) %>% wilcox_effsize(Age ~ Gender)
# Table 2
Wilcoxon.age.gender.sig <- subset(Wilcoxon.age.gender, Wilcoxon.age.gender$p < 0.05)

chi.data <- data.topic
chisq.test <- chisq.test(table(chi.data$Gender, chi.data$Topic))
chisq.test
library(vcd)
dat <- table(chi.data$Topic, chi.data$Gender)
# 1. convert the data to a table
dt <- as.table(as.matrix(dat))

# Figure 8 - mosaic plot
mosaicplot(dt, shade = T, las=1, 
           main = "Topic and Gender Association")

# In the image above, it's evident that there is a positive association between the column female and the rows topic 2, 18, 20, 24.
# There is a positive association between males and topics 3, 6, 16

##################################################
#histogram of age for each of the 3 topics with significant differences
p.age.gender.7 <- data.topic %>%
  filter(data.topic$Topic == 7) %>%
  ggplot( aes(x=Age, fill=Gender)) +
  geom_histogram(color="#e9ecef", alpha=0.6, position = 'identity') + 
  scale_fill_manual(values=c("#333333", "#cccccc")) + theme_bw() +
  scale_x_continuous(breaks = seq(0, 90, 10)) + labs(y = "Number of visits") + 
  theme(legend.position="bottom") +
  ggtitle("Topic 7 - Brachial Plexopathy (p=0.0047**)")

p.age.gender.20 <- data.topic %>%
  filter(data.topic$Topic == 20) %>%
  ggplot( aes(x=Age, fill=Gender)) + 
  geom_histogram(color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#333333", "#cccccc")) + theme_bw() +
  scale_x_continuous(breaks = seq(0, 90, 10)) + labs(y = "Number of visits") + 
  theme(legend.position="bottom") +
  ggtitle("Topic 20 - MG (p=0.0199*)")

# histogram of age for topic 6
p.age.gender.6 <- data.topic %>%
  filter(data.topic$Topic == 6) %>%
  ggplot( aes(x=Age, fill=Gender)) +
  geom_histogram(color="#e9ecef", alpha=0.6, position = 'identity') + 
  scale_fill_manual(values=c("#333333", "#cccccc")) + theme_bw() +
  scale_x_continuous(breaks = seq(0, 90, 10)) + labs(y = "Number of visits") + 
  theme(legend.position="bottom") +
  ggtitle("Topic 6 - LDPN with active EMG changes (p=0.0492*)")

library(gridExtra)
get_legend<-function(p1){
  tmp <- ggplot_gtable(ggplot_build(p1))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}
leg <- get_legend(p.age.gender.7)
library(ggpubr)
# Figure 9
ggarrange(p.age.gender.7, p.age.gender.20, p.age.gender.6,
          labels = c("A", "B", "C"),
          ncol = 3, nrow = 1,  legend = "bottom", legend.grob = leg)

####################histograms of each topic by age

## Figure 7 - age distribution by topic with topic label
ggplot(data = data.topic) +
  geom_histogram(aes(x = Age), bins = 20, colour = "black", fill = "white") +
  facet_wrap(~ topics.labeled)

