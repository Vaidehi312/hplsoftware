library("ggsankey")

df <- mtcars %>% 
  make_long(cyl, vs, am, gear, carb)

ggplot(df, aes(x = x, 
               next_x = next_x,
               node = node, 
               next_node = next_node,
               fill = factor(node),
               label = node)) +
  geom_sankey() + 
  geom_sankey_label(size = 3, color = "white", fill = "gray40") + 
  labs(x = NULL) +
  theme_sankey(base_size = 18) + 
  theme(legend.position = "none")

dummy_df <- data.frame(
  HPC = c(0, 50, 44, 67, 16, 61, 13, 48, 28, 69, 17, 34, 33, 21, 11, 47, 68, 40, 27, 5, 64, 30, 31),
  Supercluster = c(rep("Hot cohesive", 4), rep("Hot discohesive", 6), rep("Cold cohesive", 7), rep("Cold discohesive", 6)),
  GP = c("Acinar", "Solid", "Acinar", "Acinar", "Solid", "Acinar", "Solid", "Solid", "Solid", "Acinar", "Solid", "Cribriform", "Acinar", "Solid", "Acinar", "Solid", "Acinar", "Acinar", "Acinar", "Acinar", "Acinar", "Solid", "Acinar")
)



long_dummy_df <- dummy_df %>% 
  mutate(Supercluster = recode(Supercluster, 
                               "Hot discohesive" = "Hot discohesive (low risk)"))

long_dummy_df[long_dummy_df$HPC == 13, "Supercluster"] <- "Hot, discohesive (high risk)"

long_dummy_df <- long_dummy_df %>%
  make_long(Supercluster, GP)

colors <- RColorBrewer::brewer.pal(name = "Set3", n = 7)
sc_colors <- c("#F77189", "#36ADA3", "#97A431", "#A48CF4", "#FC8D62")
gp_colors <- RColorBrewer::brewer.pal(name = "Set3", n = 3)

#colors <- setNames(colors, levels(as.factor(long_dummy_df$node)))
colors <- setNames(c(sc_colors, gp_colors), c("Hot cohesive", "Hot discohesive (low risk)", "Cold cohesive", "Cold discohesive", "Hot, discohesive (high risk)",
                                              "Solid", "Acinar", "Cribriform"))

ggplot(long_dummy_df, aes(x = x, 
                          next_x = next_x,
                          node = node, 
                          next_node = next_node,
                          fill = factor(node),
                          label = node)) +
  geom_sankey(flow.alpha = .6, node.color = "gray30") + 
  geom_sankey_label(size = 4, color = "white", fill = "gray40") + 
  scale_fill_manual(values = colors) +
  labs(x = NULL) +
  scale_x_discrete(
    labels = c("Supercluster", "Growth Pattern")          
  ) +
  theme_sankey(base_size = 18) + 
  theme(legend.position = "none")
  
ggsave("SC_GP_Sankey.pdf",
       path = "./Survival",
       width = 9,
       height = 3,
       scale = 1)
