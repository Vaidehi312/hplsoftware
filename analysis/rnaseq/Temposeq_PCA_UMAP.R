library(DESeq2)
library(edgeR)
library(tidyverse)
library(umap)

load("./LUADChohort.RData")

# Parsing metadata  -----------------------------------------------------------

cluster_annotations <- read_csv('../../data/6-plex/bioclavis_pure_superclusters_split_HPC13.csv') %>% 
  column_to_rownames("...1")

## Reformatting the way cores are named to align with my HPC annotation dataframe
metadata <- LUADCohort$Metadata

reformat_core_id <- function(name) {
  parts <- unlist(strsplit(name, "-"))
  
  tma_num <- parts[1]
  row_num <- parts[2]
  col_num <- parts[3]
  
  if (nchar(tma_num) < 2) {
    new_tma_num <- paste0("0", tma_num)
  } else {
    new_tma_num <- tma_num
  }
  
  if (nchar(row_num) < 2) {
    new_row_num <- paste0("0", row_num)
  } else {
    new_row_num <- row_num
  }
  return(paste(new_tma_num, new_row_num, col_num, sep = "-"))
}
metadata$core_id <- sapply(metadata$Core, reformat_core_id)

metadata_subset <- metadata %>%
  select(c("...1", "core_id")) %>% 
  inner_join(cluster_annotations, join_by(core_id == core_ID)) %>% 
  mutate(supercluster = str_to_sentence(str_replace_all(supercluster, "_", " "))) %>% 
  column_to_rownames("...1")

metadata_subset <- metadata_subset %>% 
  mutate(supercluster = str_to_sentence(str_replace_all(supercluster, "_", " "))) %>% 
  mutate(supercluster = case_when(
    supercluster == "Hot discohesive low risk" ~ "Hot discohesive (low risk)",
    supercluster == "Hot discohesive high risk" ~ "Hot discohesive (high risk)",
    TRUE ~ supercluster
  ))

metadata_subset$supercluster <- as.factor(metadata_subset$supercluster)
  
# Normalise matrix --------------------------------------------------------

counts_subset <- LUADCohort$rawExpr %>%
  .[names(.) %in% row.names(metadata_subset)] %>%
  .[, row.names(metadata_subset)] %>%
  round(., 0) %>%
  subset(., apply(., 1, mean) >= 1) %>% # Subset genes which have mean expr >= 1
  na.omit(.) %>%
  as.matrix(.)

# dds <- DESeqDataSetFromMatrix(countData = counts_subset,
#                               colData = metadata_subset_split,
#                               design = ~ supercluster)

# vsd_full <- varianceStabilizingTransformation(dds, blind = TRUE)
# vsd <- vst(dds, blind = TRUE)
# norm_data <- assay(vsd)

y <- DGEList(counts = counts_subset)
y <- calcNormFactors(y, method = "TMM")
norm_data <- cpm(y, log = TRUE, prior.count = 1)


# PCA ---------------------------------------------------------------------

pca <- prcomp(t(norm_data), center = TRUE, scale. = TRUE)

pca_data <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  PC3 = pca$x[, 3],
  PC4 = pca$x[, 4],
  Condition = metadata_subset$supercluster
)

vars <- apply(pca$x, 2, var)
prop_x = round(vars["PC1"] / sum(vars), 4) * 100
prop_y = round(vars["PC2"] / sum(vars), 4) * 100

ggplot(pca_data, aes(x = PC1, y = PC2, color = Condition)) +
  geom_point(size = 2.5) +
  labs(x = paste("PC1 ", " (",prop_x, "%)",sep=""),
       y = paste("PC2 ", " (",prop_y, "%)",sep=""),
       title = "PCA") +
  theme_bw() +
  theme(panel.grid = element_blank())

prop_x = round(vars["PC3"] / sum(vars), 4) * 100
prop_y = round(vars["PC4"] / sum(vars), 4) * 100

ggplot(pca_data, aes(x = PC3, y = PC4, color = Condition)) +
  geom_point(size = 2.5) +
  labs(x = paste("PC3 ", " (",prop_x, "%)",sep=""),
       y = paste("PC4 ", " (",prop_y, "%)",sep=""),
       title = "PCA") +
  theme_bw() +
  theme(panel.grid = element_blank())

# 
# ggsave("PCA_Pure_cases.pdf",
#        path = "./Temposeq/Plots/Feb25_cleaned_superclusters",
#        width = 6,
#        height = 3.76,
#        units = "in",
#        scale = 0.8,
#        dpi = 300,
#        create.dir = TRUE
# )


# UMAP --------------------------------------------------------------------

umap_result <- umap(t(norm_data))

umap_data <- as.data.frame(umap_result$layout)
colnames(umap_data) <- c("UMAP1", "UMAP2")
umap_data$Condition <- metadata_subset$supercluster

ggplot(umap_data, aes(x = UMAP1, y = UMAP2)) + 
  geom_point(size = 2, aes(color = Condition)) + 
  scale_color_manual(values = c(
    "Hot cohesive"      = "#F77189",
    "Hot discohesive (low risk)"   = "#36ADA4",
    "Cold cohesive"     = "#97A431",
    "Cold discohesive"  = "#A48CF5",
    "Hot discohesive (high risk)" = "#FC8D62"
  )) +
  theme_bw() + 
  theme(panel.grid = element_blank()) + 
  labs(color = "")

ggsave("UMAP_Pure_Cases.pdf",
       path = "./",
       width = 6.5,
       height = 3.76,
       units = "in",
       scale = 0.8,
       dpi = 300,
       create.dir = TRUE)
