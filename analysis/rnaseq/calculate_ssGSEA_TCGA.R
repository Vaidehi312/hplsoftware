library(tidyverse)
library(scales)
library(ggrepel)
library(DESeq2)
library(fgsea)
library(pheatmap)
library(ComplexHeatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')
library(circlize)
library(GSVA)

tcga_tpm <- read_csv("./TCGA/TCGA-LUAD_star_symbols.csv") %>% 
  column_to_rownames("gene")

# Hallmarks

msigdb_hallmark <- loadDB("h.all.v2024.1.Hs.symbols.gmt")

ssgsea_result <- gsva(expr = as.matrix(tcga_tpm), 
                      gset.idx.list = msigdb_hallmark,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))
rownames(ssgsea_result) <- gsub("HALLMARK_", "", rownames(ssgsea_result))


Heatmap(
  ssgsea_result %>% pheatmap:::scale_rows(),
  show_column_names = FALSE
)

write.csv(ssgsea_result, file = "./TCGA/HALLMARKS_ssGSEA_TCGA.csv")

# GO BP -------------------------------------------------------------------

go_bp <- loadDB("GO_Biological_Process_2023.txt")

ssgsea_result <- gsva(expr = as.matrix(tcga_tpm), 
                      gset.idx.list = go_bp,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))

write.csv(ssgsea_result, file = "./TCGA/GO_BP_ssGSEA_TCGA.csv")

# GO CC -------------------------------------------------------------------

go_cc <- loadDB('GO_Cellular_Component_2023.txt')

ssgsea_result <- gsva(expr = as.matrix(tcga_tpm), 
                      gset.idx.list = go_cc,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))

write.csv(ssgsea_result, file = "./TCGA/GO_CC_ssGSEA_TCGA.csv")
