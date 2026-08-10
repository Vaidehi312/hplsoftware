library(MCPcounter)
library(tidyverse)
library(circlize)
library(ComplexHeatmap)

tcga_tpm <- read_csv("./TCGA/TCGA-LUAD_star_symbols.csv") %>% 
  column_to_rownames(var = "gene")

cell_estimates <- MCPcounter.estimate(tcga_tpm,
                                      featuresType = "HUGO_symbols")

write.csv(as.data.frame(cell_estimates), file = "./TCGA/MCP_Counter.csv")
