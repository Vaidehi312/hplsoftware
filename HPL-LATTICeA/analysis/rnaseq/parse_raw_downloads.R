library(tidyverse)
library(stringr)
library(biomaRt)
library(data.table)

tcga_counts <- read_tsv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD.star_counts.tsv")

ids <- str_remove(tcga_counts$Ensembl_ID, "\\..*")

tcga_counts$Ensembl_ID <- ids

mart <- useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")

map <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
               filters    = "ensembl_gene_id",
               values     = ids,
               mart       = mart)

symbol_vec <- map$hgnc_symbol[match(ids, map$ensembl_gene_id)]

tcga_sym <- tcga_counts %>% 
  mutate(gene = symbol_vec) %>% 
  filter(gene != "" & !is.na(gene)) %>% 
  group_by(gene) %>% 
  summarise(across(where(is.numeric), sum))

tcga_sym_t <- tcga_sym %>% 
  dplyr:::select(-gene) %>% 
  t(.) %>% 
  as.data.frame(.)

colnames(tcga_sym_t) <- tcga_sym$gene

tcga_sym_t <- tcga_sym_t %>% 
  rownames_to_column(var = "samples")

tcga_sym_t$samples <- substr(tcga_sym_t$samples, 1, 12)

tcga_sym_t <- tcga_sym_t %>% 
  group_by(samples) %>% 
  summarise(across(where(is.numeric), mean))

tcga_sym_tt <- tcga_sym_t %>% 
  column_to_rownames(var = "samples") %>% 
  t(.) %>% 
  as.data.frame(.)

write_csv(tcga_sym_tt %>% rownames_to_column(var = "gene"), file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD_star_symbols.csv")

###

tcga_tpm <- read_tsv("C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD.star_tpm.tsv")

ids <- str_remove(tcga_tpm$Ensembl_ID, "\\..*")

mart <- useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")

map <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
             filters    = "ensembl_gene_id",
             values     = ids,
             mart       = mart)

symbol_vec <- map$hgnc_symbol[match(ids, map$ensembl_gene_id)]

tcga_tpm_sym <- tcga_tpm %>% 
  mutate(gene = symbol_vec) %>% 
  filter(gene != "" & !is.na(gene)) %>% 
  group_by(gene) %>% 
  summarise(across(where(is.numeric), sum))

tcga_tpm_sym_t <- tcga_tpm_sym %>% 
  dplyr:::select(-gene) %>% 
  t(.) %>% 
  as.data.frame(.)

colnames(tcga_tpm_sym_t) <- tcga_tpm_sym$gene

tcga_tpm_sym_t <- tcga_tpm_sym_t %>% 
  rownames_to_column(var = "samples")

tcga_tpm_sym_t$samples <- substr(tcga_tpm_sym_t$samples, 1, 12)

tcga_tpm_sym_t <- tcga_tpm_sym_t %>% 
  group_by(samples) %>% 
  summarise(across(where(is.numeric), mean))

tcga_tpm_sym_tt <- tcga_tpm_sym_t %>% 
  column_to_rownames(var = "samples") %>% 
  t(.) %>% 
  as.data.frame(.)

write_csv(tcga_tpm_sym_tt %>% rownames_to_column(var = "gene"), file = "C:\\Users\\krakovic\\OneDrive - University of Glasgow\\Temposeq\\TCGA_20250424\\TCGA-LUAD_star_tpm_symbols.csv")



