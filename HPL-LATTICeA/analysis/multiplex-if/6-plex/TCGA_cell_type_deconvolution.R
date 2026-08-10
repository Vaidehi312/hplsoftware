library(MCPcounter)
library(tidyverse)
library(circlize)
library(ComplexHeatmap)

tcga_tpm <- read_csv("./TCGA-LUAD_star_tpm_symbols.csv") %>% 
  column_to_rownames(var = "gene")

tcga_malignant <- read_csv("./tcga_agg_by_sum_5sc.csv") %>% 
  dplyr:::select(c("samples", "Hot, cohesive", "Hot, discohesive", "HPC 13", "Cold, discohesive", 'Cold, cohesive')) %>% 
  filter(., samples %in% colnames(tcga_tpm)) %>% 
  arrange(samples) %>% 
  column_to_rownames(., var = "samples")

tcga_cluster_labs <- as.data.frame(names(tcga_malignant))
colnames(tcga_cluster_labs) <- c("Supercluster")

###

cell_estimates <- MCPcounter.estimate(tcga_tpm,
                                      featuresType = "HUGO_symbols")

cell_estimates_hpl <- as.data.frame(cell_estimates) %>%
  dplyr:::select(rownames(tcga_malignant))

mcp_corr <- cor(tcga_malignant,
                t(cell_estimates_hpl),
                method = "spearman")

rownames(mcp_corr) <- c("Hot cohesive", "Hot discohesive (low risk)", "Hot discohesive (high risk)", "Cold discohesive", "Cold cohesive")

col_colors <- colorRamp2(c(-0.4, 0, 0.4), c("#4E79A7", "white", "#E15759"))

p <- Heatmap(mcp_corr,
        col = col_colors,
        border = TRUE,
        name = "Spearman\ncorrelation",
        row_dend_width = unit(4, "mm"),
        column_dend_height = unit(4, "mm"))  

pdf(file="./TCGA_MCPCOUNTER_250424.pdf", width = 5, height = 3)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 2), "mm"))
dev.off()

###

compute_correlations_with_pvals <- function(matrix1, matrix2, corr_method) {
  # Initialize matrices for correlations and p-values
  n_rows <- ncol(matrix1)
  n_cols <- ncol(matrix2)
  cors <- matrix(NA, nrow = n_rows, ncol = n_cols)
  pvals <- matrix(NA, nrow = n_rows, ncol = n_cols)
  
  # Compute correlations and p-values
  for (i in 1:n_rows) {
    for (j in 1:n_cols) {
      test_result <- cor.test(matrix1[,i], matrix2[,j], method = corr_method)
      cors[i,j] <- test_result$estimate
      pvals[i,j] <- test_result$p.value
    }
  }
  
  # Adjust p-values for multiple testing
  # adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows)
  adjusted_pvals <- matrix(p.adjust(pvals, method = "BH"), nrow = n_rows, ncol = n_cols)
  
  return(list(correlations = cors, 
              raw_pvals = pvals, 
              adjusted_pvals = adjusted_pvals))
}

stat_results <- compute_correlations_with_pvals(t(cell_estimates_hpl), tcga_malignant, corr_method = "spearman")
stat_cor <- stat_results$correlations
colnames(stat_cor) <- c("Hot cohesive", "Hot discohesive\n(low risk)", "Hot discohesive\n(high risk)", "Cold discohesive", "Cold cohesive")
rownames(stat_cor) <- rownames(cell_estimates_hpl)

sig_matrix <- matrix("", nrow = nrow(stat_results$adjusted_pvals), ncol = ncol(stat_results$adjusted_pvals))
sig_matrix[stat_results$adjusted_pvals < 0.001] <- "***"
sig_matrix[stat_results$adjusted_pvals < 0.01 & stat_results$adjusted_pvals >= 0.001] <- "**"
sig_matrix[stat_results$adjusted_pvals < 0.05 & stat_results$adjusted_pvals >= 0.01] <- "*"

p <- Heatmap(
  stat_cor,
  name = "Spearman\ncorrelation",
  col = col_colors,
  border = TRUE,
  row_names_side = "left",
  row_dend_side = "right", 
  cell_fun = function(j, i, x, y, width, height, fill) {
    if (sig_matrix[i, j] != "") {
      grid.text(sig_matrix[i, j], x, y)
    }
  }
)

pdf(file="./TCGA_MCPCOUNTER_250424_stats.pdf", width = 4.6, height = 4)
draw(p, heatmap_legend_side = "left", annotation_legend_side = "left", padding = unit(c(2, 2, 2, 2), "mm"))
dev.off()

