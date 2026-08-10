library(tidyverse)
library(GSVA)
library(circlize)
library(pheatmap)
library(ComplexHeatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')
library(edgeR)

load("./LUADChohort.RData")

# Preprocessing count data ------------------------------------------------

raw_counts <- LUADCohort$rawExpr
y <- DGEList(counts = raw_counts)
y <- calcNormFactors(y, method = "TMM")
logCPM <- cpm(y, log = TRUE, prior.count = 1)

# Hallmarks

msigdb_hallmark <- loadDB("h.all.v2024.1.Hs.symbols.gmt")

ssgsea_result <- gsva(expr = as.matrix(logCPM), 
                      gset.idx.list = msigdb_hallmark,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))
rownames(ssgsea_result) <- gsub("HALLMARK_", "", rownames(ssgsea_result))


Heatmap(
  ssgsea_result %>% pheatmap:::scale_rows(),
  show_column_names = FALSE
)

write.csv(ssgsea_result, file = "./HALLMARKS_ssGSEA_LATTICeA.csv")

# KEGG --------------------------------------------------------------------

kegg <- loadDB("KEGG_2021_Human.txt")

ssgsea_result <- gsva(expr = logCPM, 
                      gset.idx.list = kegg,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))

Heatmap(
  ssgsea_result %>% pheatmap:::scale_rows(),
  show_column_names = FALSE
)

write.csv(ssgsea_result, file = "./KEGG_ssGSEA_LATTICeA.csv")


# GO BP -------------------------------------------------------------------

go_bp <- loadDB("GO_Biological_Process_2023.txt")

ssgsea_result <- gsva(expr = logCPM, 
                      gset.idx.list = go_bp,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))

Heatmap(
  ssgsea_result %>% pheatmap:::scale_rows(),
  show_column_names = FALSE
)

write.csv(ssgsea_result, file = "./GO_BP_ssGSEA_LATTICeA.csv")

# GO CC -------------------------------------------------------------------

go_cc <- loadDB('GO_Cellular_Component_2023.txt')

ssgsea_result <- gsva(expr = logCPM, 
                      gset.idx.list = go_cc,
                      method = 'ssgsea',
                      BPPARAM = MulticoreParam(workers = detectCores() - 1))

Heatmap(
  ssgsea_result %>% pheatmap:::scale_rows(),
  show_column_names = FALSE
)

write.csv(ssgsea_result, file = "./GO_CC_ssGSEA_LATTICeA.csv")
