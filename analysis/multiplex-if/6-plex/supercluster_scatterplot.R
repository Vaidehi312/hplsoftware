library(tidyverse)
library(ggrepel)

annot_stats <- read.csv("./merged_cluster_lym_log10_density_and_noise.csv")

ggplot(annot_stats, aes(x = robust_z_score_x, y = robust_z_score_y)) + 
  geom_point(aes(color = Supercluster), size=3) + 
  labs(x = "Discohesion score", y = "Lymphocyte density score") + 
  scale_color_manual(values = c(
    "Hot cohesive"      = "#F77189",
    "Hot discohesive"   = "#36ADA4",
    "Cold cohesive"     = "#97A431",
    "Cold discohesive"  = "#A48CF5"
  )) +
  geom_text_repel(data = annot_stats, aes(label = HPC), point.size = 3, size = 3, max.time = 10, max.iter = 5000) + 
  theme_bw() + 
  theme(panel.grid = element_blank(),
        legend.position = "none",
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

ggsave("supercluster_scatterplot.pdf",
       path = "./figures",
       scale = 0.7,
       dpi = 300
)
