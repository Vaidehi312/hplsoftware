library(tidyverse)
library(scales)
library(ggrepel)
library(DESeq2)
library(fgsea)
library(pheatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')
library(rstatix)
library(ggpubr)
library(ComplexHeatmap)
library(circlize)

load("./LUADChohort.RData")

hallmarks <- loadDB("h.all.v2024.1.Hs.symbols.gmt")
go_bp <- loadDB("GO_Biological_Process_2023.txt")
kegg <- loadDB("KEGG_2021_Human.txt")

cluster_labs <- read_csv('../../data/6-plex/merged_cluster_lym_log10_density_and_noise.csv') %>% 
  select(c("HPC", "Supercluster"))

cluster_annotations <- read_csv('../../data/6-plex/bioclavis_pure_superclusters_split_HPC13.csv') %>% 
  column_to_rownames("...1")

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
metadata[, c("Core", "core_id")]

metadata_subset = metadata[, c("...1", "core_id")] %>%
  merge(., cluster_annotations, by.x=2, by.y=1) %>%
  column_to_rownames((var = '...1'))

metadata_subset <- metadata_subset %>% 
  mutate(., supercluster = str_to_sentence(str_replace_all(supercluster, "_", " ")))

# Cold cohesive vs. Cold discohesive --------------------------------------

all_cold <- c("Cold cohesive", "Cold discohesive")
cold_subset <- metadata_subset[metadata_subset$supercluster %in% all_cold, ]

cold_raw_counts <- LUADCohort$rawExpr %>% 
  select(row.names(cold_subset))

cold_raw_counts <- cold_raw_counts[, row.names(cold_subset)]

cold_raw_counts_filtered_genes <- round(cold_raw_counts, 0) %>%
  subset(., apply(cold_raw_counts, 1, mean) >= 1) %>% # Subset genes which have mean expr >= 1
  na.omit(.)

cold_raw_counts_matrix <- as.matrix(cold_raw_counts_filtered_genes)
cold_subset$supercluster <- factor(cold_subset$supercluster)

cold_dds <- DESeqDataSetFromMatrix(countData = cold_raw_counts_matrix, colData = cold_subset, design = ~ supercluster)
cold_dds <- DESeq(cold_dds)

cold_resdata <- data.frame(round(counts(cold_dds, normalized = TRUE), 2))
colnames(cold_resdata) <- colnames(cold_raw_counts_matrix)

cold_de <- data.frame(results(cold_dds, c("supercluster", "Cold discohesive", "Cold cohesive"))) %>% 
  arrange(padj)

#cold_de <- cold_de[order(cold_de[, 'padj'], decreasing = FALSE), ]

cold_de$mlog10p <- -log(cold_de$padj, 10)

sig_up <- cold_de %>% 
  filter(log2FoldChange > 1 & padj < 0.01)

sig_down <- cold_de %>% 
  filter(log2FoldChange < -1 & padj < 0.01)

ggplot(cold_de, aes(x = log2FoldChange, y = mlog10p)) +
  geom_point(aes(color = "Not Significant")) +
  geom_point(data = sig_up, aes(color = "Up in cold\ndiscohesive")) +
  geom_point(data = sig_down, aes(color = "Down in cold\ndiscohesive")) +
  scale_color_manual(values = c(
    "Not Significant" = "grey",
    "Up in cold\ndiscohesive" = "#E15759",
    "Down in cold\ndiscohesive" = "#4E79A7"
  ), name = "") +
  geom_text_repel(data = sig_up %>% slice_head(n = 10), aes(label = row.names(sig_up %>% slice_head(n = 10))), max.overlaps = 40) + 
  geom_text_repel(data = sig_down %>% slice_head(n = 10), aes(label = row.names(sig_down %>% slice_head(n = 10))), max.overlaps = 40) + 
  labs(x = 'L2FC', y = '-log10p', title = 'DE Cold, discohesive vs. Cold, cohesive') +
  theme_bw() + 
  theme(panel.grid = element_blank())

ggsave("DE_COLD_VOLC.pdf",
       path = "./",
       scale = 0.4,
       width = 13,
       height = 9.5,
       dpi = 300
)


# Rank genes for GSEA

cold_ranked_genes <- cold_de$log2FoldChange
names(cold_ranked_genes) <- row.names(cold_de)
cold_ranked_genes <- sort(cold_ranked_genes, decreasing = TRUE)

# Hallmarks

cold_gsea_results_hallmarks <- fgsea(
  pathways = hallmarks,
  stats = cold_ranked_genes,      
  minSize = 5,
  maxSize = length(cold_ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(cold_gsea_results_hallmarks, padj < 0.01) %>%
  arrange(padj) %>%
  # arrange(desc(abs(NES))) %>%
  mutate(pathway = sub("^HALLMARK_", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 60)) %>% 
  slice_head(n = 10)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Enriched in Cold discohesive vs. Cold cohesive", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

ggsave("DE_COLD_GSEA_HALLMARKS_p001.pdf",
       path = "./",
       scale = 0.4,
       width = 15,
       height = 7,
       dpi = 600
)

# KEGG

cold_gsea_results_hallmarks <- fgsea(
  pathways = kegg,
  stats = cold_ranked_genes,      
  minSize = 5,
  maxSize = length(cold_ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(cold_gsea_results_hallmarks, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Enriched in Cold discohesive vs. Cold cohesive", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

ggsave("DE_COLD_GSEA_KEGG_p001.pdf",
       path = "./",
       scale = 0.5,
       width = 16,
       height = 11,
       dpi = 300
)

# GO CC

go_cc <- loadDB('GO_Cellular_Component_2023.txt')

cold_gsea_results_gocc <- fgsea(
  pathways = go_cc,
  stats = cold_ranked_genes,      
  minSize = 5,
  maxSize = length(cold_ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(cold_gsea_results_gocc, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = gsub("\\(.*?\\)", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 10)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Enriched in Cold discohesive vs. Cold cohesive", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

ggsave("DE_COLD_GSEA_GO_CC_001_n10.pdf",
       path = "./",
       scale = 0.4,
       width = 15,
       height = 7,
       dpi = 300
)

cold_ecm_genes <- cold_de %>% 
  # filter(rownames(.) %in% go_cc$`Collagen-Containing Extracellular Matrix (GO:0062023)`) %>% 
  # filter(rownames(.) %in% go_cc$`Focal Adhesion (GO:0005925)`) %>% 
  # filter(rownames(.) %in% go_cc$`Cell-Substrate Junction (GO:0030055)`) %>% 
  filter(rownames(.) %in% go_cc$`Keratin Filament (GO:0045095)`) %>% 
  dplyr:::select(log2FoldChange) %>% 
  arrange(desc(abs(log2FoldChange))) %>% 
  slice_head(n = 30)

cold_resdata_genes <- cold_resdata %>% 
  # filter(rownames(.) %in% go_cc$`Collagen-Containing Extracellular Matrix (GO:0062023)`) %>% 
  # filter(rownames(.) %in% go_cc$`Focal Adhesion (GO:0005925)`) %>% 
  # filter(rownames(.) %in% go_cc$`Cell-Substrate Junction (GO:0030055)`) %>% 
  filter(rownames(.) %in% rownames(cold_ecm_genes))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))
column_ha <- HeatmapAnnotation(
  supercluster = cold_subset$supercluster,
  col = list(supercluster = c("Cold discohesive" = "hotpink1",
                              "Cold cohesive" = "dodgerblue")))


Heatmap(as.matrix(cold_resdata_genes) %>% pheatmap:::scale_rows(),
        col = col_colors, 
        top_annotation = column_ha,
        cluster_columns = cluster_within_group(cold_resdata_genes, cold_subset$supercluster))

tidy_resdata_genes <- as.data.frame(t(cold_resdata_genes %>% rownames_to_column(var = "samples")))
colnames(tidy_resdata_genes) <- tidy_resdata_genes["samples", ]
tidy_resdata_genes <- tidy_resdata_genes %>% 
  filter(!row_number() %in% c(1)) %>% 
  rownames_to_column(var = "samples") %>% 
  pivot_longer(., cols = rownames(cold_resdata_genes), names_to = "gene", values_to = "expr") %>% 
  left_join(cold_subset %>% rownames_to_column(var = "samples") %>% select(c("samples", "supercluster")), join_by(samples == samples)) %>% 
  mutate(expr = as.numeric(expr))


ggplot(tidy_resdata_genes %>% filter(gene == "KRT6A"), 
       aes(x = supercluster, y = expr)) +
  geom_boxplot() + 
  scale_y_log10() + 
  stat_compare_means(method = "wilcox.test")

norm_genes <- as.data.frame(LUADCohort$normExpr) %>% 
  filter(rownames(.) %in% rownames(cold_resdata_genes)) %>% 
  dplyr:::select(rownames(metadata_subset))

norm_genes <- norm_genes[, rownames(metadata_subset)]

metadata_subset$supercluster <- as.factor(metadata_subset$supercluster)

column_ha <- HeatmapAnnotation(df = metadata_subset %>% select(supercluster))

Heatmap(as.matrix(norm_genes) %>% pheatmap:::scale_rows(),
        top_annotation = column_ha)
        # cluster_columns = cluster_within_group(as.matrix(norm_genes), metadata_subset$supercluster))


# GO BP

cold_gsea_results_gobp <- fgsea(
  pathways = go_bp,
  stats = cold_ranked_genes,      
  minSize = 5,
  maxSize = length(cold_ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(cold_gsea_results_gobp, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = gsub("\\(.*?\\)", "", pathway)) %>% 
  # mutate(pathway = str_wrap(pathway, width = 60)) %>% 
  slice_head(n = 10)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Enriched in Cold discohesive vs. Cold cohesive", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

ggsave("DE_COLD_GSEA_GO_BP_p001.pdf",
       path = "./",
       scale = 0.4,
       width = 20,
       height = 8,
       dpi = 300
)

