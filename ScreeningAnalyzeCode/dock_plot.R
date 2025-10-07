library(ggplot2)
library(reportROC)

######## docking predicted results (incl. docking, similarity score, probability)
dock = read.csv("./instaDock_result/docking_pred_binding.csv")
top = dock[dock$mark == 1, ]
bottom = dock[dock$mark == 0, ]

# determine optimal cut-off
reportROC(gold = dock$mark, predictor = dock$pKi,plot=TRUE)


### plot of similarity score vs. docking pKi (grouped by mark)
ggplot(dock, aes(x = Similarity.Scores, y = pKi, color = as.factor(mark))) +
  geom_point() +
  scale_color_manual(values=c("#009E73","#D55E00")) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  labs(color = "class") 



dock_top <- ggplot(top, aes(x = Similarity.Scores, y = pKi, color=as.factor(mark))) +
              geom_point() +
              scale_color_manual(values=c("#009E73")) +
              scale_x_continuous(
                breaks = c(seq(0, 0.14, by = 0.04)),
                labels = c(seq(0, 0.14, by = 0.04)),
                limits = c(0, 0.14)
              ) +
              scale_y_continuous(
                breaks = c(seq(0, 12, by = 2)),
                labels = c(seq(0, 12, by = 2)),
                limits = c(0, 12)
              ) +
              theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                    panel.background = element_blank(), axis.line = element_line(colour = "black")) +
              theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_top.png", dock_top, width = 6, height = 6, dpi = 300)
  

dock_bottom <-ggplot(bottom, aes(x = Similarity.Scores, y = pKi, color = as.factor(mark))) +
                geom_point() +
                scale_color_manual(values=c("#D55E00")) +
                scale_x_continuous(
                  breaks = c(seq(7, 10.5, by = 0.5)),
                  labels = c(seq(7, 10.5, by = 0.5)),
                  limits = c(7, 10.5)
                ) +
                scale_y_continuous(
                  breaks = c(seq(0, 4, by = 2)),
                  labels = c(seq(0, 4, by = 2)),
                  limits = c(0, 12)
                ) +
                theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                      panel.background = element_blank(), axis.line = element_line(colour = "black")) +
                theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_bottom.png", dock_bottom, width = 6, height = 6, dpi = 300)



########
## set color class
library(scales)

classes=8

#set margins of plot area
par(mai = c(0.1, 0, 0.1, 0), bg = "grey85")

#create plot with ggplot2 default colors from 1 to 8
gc.grid <- layout(matrix(1:classes, nrow = classes))
for(i in classes:classes){
  gc.ramp <- hue_pal()(i)
  plot(c(0, classes), c(0,1),
       type = "n", 
       bty="n", 
       xaxt="n", 
       yaxt="n", xlab="", ylab="")
  for(j in 1:i){
    rect(j - 1, 0, j - 0.25, 1, col = gc.ramp[j])
    print(gc.ramp[j])
  }
}








### plot of Similarity(compared w Prob) vs. docking pKi (grouped by uniprot ID)
ggplot(dock, aes(x = Similarity.Scores, y = pKi, color = Uniprot_ID)) +
  geom_point() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  labs(color = "Uniprot ID") 


dock_top <- ggplot(top, aes(x = Similarity.Scores, y = pKi, color=Uniprot_ID)) +
  geom_point() +
  scale_x_continuous(
    breaks = c(seq(0, 0.14, by = 0.04)),
    labels = c(seq(0, 0.14, by = 0.04)),
    limits = c(0, 0.14)
  ) +
  scale_y_continuous(
    breaks = c(seq(0, 12, by = 2)),
    labels = c(seq(0, 12, by = 2)),
    limits = c(0, 12)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_top_by_Uniprot.png", dock_top, width = 6, height = 6, dpi = 300)


dock_bottom <- ggplot(bottom, aes(x = Similarity.Scores, y = pKi, color = Uniprot_ID)) +
  geom_point() +
  scale_color_manual(values=c("#F8766D","#00BE67","#FF61CC")) +
  scale_x_continuous(
    breaks = c(seq(7, 10.5, by = 0.5)),
    labels = c(seq(7, 10.5, by = 0.5)),
    limits = c(7, 10.5)
  ) +
  scale_y_continuous(
    breaks = c(seq(0, 4, by = 2)),
    labels = c(seq(0, 4, by = 2)),
    limits = c(0, 12)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_bottom_by_Uniprot.png", dock_bottom, width = 6, height = 6, dpi = 300)



dock_top_prob <- ggplot(top, aes(x = Probabilities, y = pKi, color = Uniprot_ID)) +
  geom_point() +
  scale_x_continuous(
    breaks = c(seq(0.75, 0.9, by = 0.05)),
    labels = c(seq(0.75, 0.9, by = 0.05)),
    limits = c(0.75, 0.9)
  ) +
  scale_y_continuous(
    breaks = c(seq(0, 4, by = 2)),
    labels = c(seq(0, 4, by = 2)),
    limits = c(0, 12)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_top_prob_by_Uniprot.png", dock_top_prob, width = 6, height = 6, dpi = 300)



dock_bottom_prob <- ggplot(bottom, aes(x = Probabilities, y = pKi, color = Uniprot_ID)) +
  geom_point() +
  scale_color_manual(values=c("#F8766D","#00BE67","#FF61CC")) +
  scale_x_continuous(
    breaks = c(seq(0.3, 0.55, by = 0.05)),
    labels = c(seq(0.3, 0.55, by = 0.05)),
    limits = c(0.3, 0.55)
  ) +
  scale_y_continuous(
    breaks = c(seq(0, 12, by = 2)),
    labels = c(seq(0, 12, by = 2)),
    limits = c(0, 12)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=16))
ggsave("./plot/dock_bottom_prob_by_Uniprot.png", dock_bottom_prob, width = 6, height = 6, dpi = 300)







############ SPR
### MET
Cycle = c(seq(8,20), 21)
RU = c(0, 1.025, 1.091, 6.981, 125.7, 11.9, 1.081, 1.215, 1.152, 7.209, 0.8102, 10.63, 118.4, 0)
group = c('Control Sample',  rep(('Sample'),12), ('Control Sample'))
df <- data.frame(Cycle, RU, group)

SPR_MET <- ggplot(df, aes(x = Cycle, y = RU, color=group)) +
  geom_point(size=4) +
  geom_hline(yintercept=0) +
  scale_x_continuous(
    breaks = c(seq(-5, 25, by = 5)),
    labels = c(seq(-5, 25, by = 5)),
    limits = c(-5, 25)
  ) +
  scale_y_continuous(
    breaks = c(seq(-20, 140, by = 20)),
    labels = c(seq(-20, 140, by = 20)),
    limits = c(-20, 140)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=12))

ggsave("./plot/SPR_MET.png", SPR_MET, width = 6, height = 3.5, dpi = 300)



### CDK4
Cycle = c(seq(7,14), 16)
RU = c(0, 6.575, 11.5, 6.772, 0.8821, 8.173, 7.342, 348.9, 0)
group = c('Control Sample',  rep(('Sample'),7), ('Control Sample'))
df <- data.frame(Cycle, RU, group)

SPR_CDK4 <- ggplot(df, aes(x = Cycle, y = RU, color=group)) +
  geom_point(size=4) +
  geom_hline(yintercept=0) +
  scale_x_continuous(
    breaks = c(seq(-2, 18, by = 2)),
    labels = c(seq(-2, 18, by = 2)),
    limits = c(-2, 18)
  ) +
  scale_y_continuous(
    breaks = c(seq(-50, 400, by = 50)),
    labels = c(seq(-50, 400, by = 50)),
    limits = c(-50, 400)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=12))

ggsave("./plot/SPR_CDK4.png", SPR_CDK4, width = 6, height = 3.5, dpi = 300)



### PGFRA
Cycle = c(seq(7,13), 15,17)
RU = c(0, 0.8703, 0.7929, 0.5853, 0.608, 0.801, 5.148, 15.76, 0)
group = c('Control Sample',  rep(('Sample'),7), ('Control Sample'))
df <- data.frame(Cycle, RU, group)

SPR_PGFRA <- ggplot(df, aes(x = Cycle, y = RU, color=group)) +
  geom_point(size=4) +
  geom_hline(yintercept=0) +
  scale_x_continuous(
    breaks = c(seq(-2, 18, by = 2)),
    labels = c(seq(-2, 18, by = 2)),
    limits = c(-2, 18)
  ) +
  scale_y_continuous(
    breaks = c(seq(-2, 18, by = 2)),
    labels = c(seq(-2, 18, by = 2)),
    limits = c(-2, 18)
  ) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(legend.position = "none", axis.text=element_text(size=12))

ggsave("./plot/SPR_PGFRA.png", SPR_PGFRA, width = 6, height = 3.5, dpi = 300)
