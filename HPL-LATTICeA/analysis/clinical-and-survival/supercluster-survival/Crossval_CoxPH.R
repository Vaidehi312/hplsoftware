library(survminer)
library(tidyverse)
library(survival)

lattice_data <- read_csv("./complete_sample_rep_latticea_clin_var.csv") %>%
  column_to_rownames("...1") %>%
  rename(`Hot, discohesive (high risk)` = `HPC 13`) %>%
  rename(`Hot discohesive (low risk)` = `Hot, discohesive`) %>% 
  rename(Age = age)

lattice_data$Stage <- factor(
  max.col(lattice_data[, c("stage_1", "stage_2", "stage_3")]),
  levels = 1:3,
  labels = c("Stage I", "Stage II", "Stage III/IV")
)

lattice_data$`IASLC Grade` <- factor(
  lattice_data$iaslc_grade_,
  levels = c("G1", "G2", "G3"),
  labels = c("Grade 1", "Grade 2", "Grade 3")
)

lattice_data$Sex <- factor(lattice_data$male,
                           levels = 0:1,
                           labels = c("Female", "Male"))

lattice_data <- lattice_data %>%
  select(-c(stage_1, stage_2, stage_3, iaslc_grade_, male))

tcga_data <- read_csv('./additional_sample_rep_tcga_clin_var.csv') %>%
  column_to_rownames("...1") %>%
  rename(`Hot, discohesive (high risk)` = `HPC 13`) %>%
  rename(`Hot discohesive (low risk)` = `Hot, discohesive`) %>% 
  rename(Age = age)

tcga_data$Stage <- factor(
  max.col(tcga_data[, c("stage_1", "stage_2", "stage_3")]),
  levels = 1:3,
  labels = c("Stage I", "Stage II", "Stage III/IV")
)

tcga_data$`IASLC Grade` <- factor(
  tcga_data$iaslc_grade_,
  levels = c("G1", "G2", "G3"),
  labels = c("Grade 1", "Grade 2", "Grade 3")
)

tcga_data$Sex <- factor(tcga_data$male,
                        levels = 0:1,
                        labels = c("Female", "Male"))

tcga_data <- tcga_data %>%
  select(-c(stage_1, stage_2, stage_3, iaslc_grade_, male))


set.seed(8)
folds <- sample(rep(1:5, length.out = nrow(lattice_data)))

results <- data.frame(fold = integer(),
                      cindex_train = numeric(),
                      cindex_test = numeric(),
                      cindex_external = numeric())


for(i in 1:5){
  train <- lattice_data %>% filter(folds != i)
  test  <- lattice_data %>% filter(folds == i)
  
  cox_fit <- coxph(Surv(os_event_data, os_event_ind) ~ .,
                   data = train %>% select(-samples),
                   ties = "exact")
  
  fp <- ggforest(cox_fit, 
           data = train)
  
  print(fp)
  
  ggsave(paste0("All_clin_Superclusters_only_fold_", i, "_.pdf"),
         plot = fp,
         scale = 0.7,
         width = 12,
         height = 6,
         path = "./Survival")
  
  cox_summary_df <- broom::tidy(
    cox_fit,
    exponentiate = TRUE,   
    conf.int     = TRUE    
  )
  
  write_csv(cox_summary_df, file = paste0("./Survival/cox_summary_clin_Superclusters_fold_", i, "_.csv"))

  lp_train <- predict(cox_fit, newdata = train, type = "lp") * -1
  ci_train <- concordance(Surv(train$os_event_data, train$os_event_ind) ~ lp_train)$concordance
  
  lp_test <- predict(cox_fit, newdata = test, type = "lp") * -1
  ci_test <- concordance(Surv(test$os_event_data, test$os_event_ind) ~ lp_test)$concordance
  
  lp_external <- predict(cox_fit, newdata = tcga_data, type = "lp") * -1
  ci_external <- concordance(Surv(tcga_data$os_event_data, tcga_data$os_event_ind) ~ lp_external)$concordance
  
  results <- rbind(results, data.frame(fold = i,
                                       cindex_train = ci_train,
                                       cindex_test = ci_test,
                                       cindex_external = ci_external))
}

results_long <- results %>%
  pivot_longer(cols = c("cindex_train", "cindex_test", "cindex_external"),
               names_to = "dataset", values_to = "cindex")

results_long$dataset <- recode(results_long$dataset,
                               "cindex_train" = "Train",
                               "cindex_test" = "Test",
                               "cindex_external" = "External")

cindex_plot <- ggplot(results_long, aes(x = factor(fold), y = cindex, fill = dataset)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Fold",
       y = "Concordance Index",
       title = "C-Index across Folds for Train, Test, and External Datasets") +
  theme_minimal()

results_long <- results %>%
  pivot_longer(cols = c("cindex_train", "cindex_test", "cindex_external"),
               names_to = "dataset", values_to = "cindex")

results_long$dataset <- recode(results_long$dataset,
                               "cindex_train" = "Train",
                               "cindex_test" = "Test",
                               "cindex_external" = "TCGA")
results_long$dataset <- factor(results_long$dataset, levels = c("Train", "Test", "TCGA"))

cindex_plot <- ggplot(results_long, aes(x = dataset, y = cindex)) +
  stat_boxplot(geom= 'errorbar' , width = 0.3, position = position_dodge(width = 0.1), coef = Inf) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +  
  geom_jitter(width = 0.1, size = 3, aes(color = dataset)) +
  labs(x = "Dataset",
       y = "C-Index",
       title = "") +
  scale_y_continuous(
    limits = c(0.62, 0.72),
    breaks = seq(0.62, 0.72, by = 0.02)
  ) +
  theme_bw() +  
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(colour = "black"),
        axis.text.y = element_text(colour = "black"),
        legend.position = "none")

print(cindex_plot)

ggsave("All_clin_Superclusters_only_cindex.pdf",
       path = "./Survival",
       plot = cindex_plot,
       scale = 0.5,
       width = 4,
       height = 6)

write_csv(results_long, file = "./Survival/c_index_sc_clin_all_folds.csv")
