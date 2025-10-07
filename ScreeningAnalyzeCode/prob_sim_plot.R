library(ggplot2)




#### AI-Bind
pred_aibind = read.csv("D:/computational_biology/gitProject/AI-Bind-v1.1/root/sars_preidcitons_unseen_nodes.csv")


prob_aibind <- ggplot(pred_aibind, aes(x = ID, y = Averaged.Predictions, color = ID)) +
  geom_point(shape=4, alpha=0.05) + 
  theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank())
ggsave("./plot/prob_aibind_ID.png", prob_aibind, width = 6, height = 6, dpi = 300)






### plot of similarity score vs. probability
pred = read.csv("screening_ZINC20.csv")
ggplot(pred, aes(x = Similarity.Scores, y = Probabilities, color = Uniprot_ID)) +
  geom_point(shape=4, alpha=0.05) +
  theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
ggsave("./plot/prob_aibind_ID.png", prob_aibind, width = 6, height = 6, dpi = 300)



# randomly select obs equally for each uniprot_ID
library(dplyr)

pred_all = read.csv("./screening_ZINC20_all_pairs.csv")
sampled_pred <- pred_all %>%
  group_by(Uniprot_ID) %>%
  sample_n(10000)

scores_vs_prob <- ggplot(sampled_pred, aes(x = Similarity.Scores, y = Probabilities, color = Uniprot_ID)) +
  geom_point(shape=4, alpha=0.05) +
  theme(legend.position = "none", axis.text=element_text(size=16)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
ggsave("scores_vs_prob.png", scores_vs_prob, width = 6, height = 4, dpi = 300)


#######
#
scores_vs_prob_ours = ggplot(sampled_pred_BACPI, aes(x = Similarity.Scores, y = Probabilities.y, color = Uniprot_ID)) +
  geom_point(shape=4, alpha=0.05)
ggsave("scores_vs_prob_ours.png", scores_vs_prob_ours, width = 6, height = 4, dpi = 300)



library(beeswarm)
# Bee swarm plot for probabilties and scores by targets
beeswarm(pred$Probabilities ~ pred$Uniprot_ID)
beeswarm(pred$Similarity.Scores ~ pred$Uniprot_ID, pch = 19)



human = read.csv("D:/projects/BACPI_compare/Data/human.txt", sep=' ', header=F)
human$target <- as.integer(factor(human$V2, levels = unique(human$V2)))


# pred from BACPI
pred_all_BACPI = read.csv("D:/projects/BACPI_compare/result/screening_ZINC20_all_pairs.csv")
x = merge(pred_all_BACPI, pred_all, by=c('Ligand.SMILES','PDB_ID','het_name','Uniprot_ID','Screening.SMILES'),
          all.y=F)

sampled_pred_BACPI <- x %>%
  group_by(Uniprot_ID) %>%
  sample_n(10000)


df <- df %>%
  mutate(percentile = ntile(value, 100))

# Select observations based on percentiles
selected_df <- df %>%
  filter(percentile %in% c(100, 90, 75, 50, 25, 0)) %>%
  group_by(percentile) %>%
  sample_n(100)

#
scores_vs_prob_BACPI = ggplot(sampled_pred_BACPI, aes(x = Similarity.Scores, y = Probabilities.x, color = Uniprot_ID)) +
  geom_point(shape=4, alpha=0.05)
ggsave("scores_vs_prob_BACPI.png", scores_vs_prob_BACPI, width = 6, height = 4, dpi = 300)