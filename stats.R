library(dplyr)

dataGT = read.csv("statsGT.csv")
summary(dataGT)

dataROI = read.csv("statsROI.csv")
summary(dataROI)

dataScored1 = read.csv("data/BUSL_Rubric_PublicAccess_rle2021_2021-04-28.csv")
summary(dataEllis)

dataScored = read.csv("dataScored.csv")
summary(dataScored)

prepare_data30_2 <- function(df){
  data30 = df[,2:31]
  cat_cols = c("type_of_breast_surgery", "cancer_type_detailed", "cellularity", "chemotherapy")
  cat_cols = c(cat_cols, c("pam50_._claudin.low_subtype","cohort", "er_status_measured_by_ihc", "er_status", "neoplasm_histologic_grade"))
  cat_cols = c(cat_cols, c("her2_status_measured_by_snp6","her2_status", "tumor_other_histologic_subtype", "hormone_therapy", "inferred_menopausal_state"))
  cat_cols = c(cat_cols, c("integrative_cluster", "oncotree_code", "pr_status"))
  cat_cols = c(cat_cols, c("radio_therapy", "death_from_cancer"))
  # cols = c(cols, c("cancer_type","tumor_stage","X3.gene_classifier_subtype","primary_tumor_laterality", "overall_survival", "", "", ""))
  num_cols = c("age_at_diagnosis", "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index", "overall_survival_months", "tumor_size")
  
  data30 = data30 %>%
    select(age_at_diagnosis:death_from_cancer) %>%
    select(!c(tumor_stage, cancer_type, primary_tumor_laterality, X3.gene_classifier_subtype, overall_survival)) %>%
    filter(death_from_cancer != "Living") %>%
    mutate(across(where(is.character), ~na_if(., " "))) %>%
    mutate(across(where(is.character), ~na_if(., ""))) %>%
    na.omit() %>%
    mutate_at(.vars = cat_cols, .funs = ~as.factor(.)) %>%
    mutate_at(.vars = num_cols, .funs = ~scale(.))
  return(data30)
}

prepare_data <- function(df1, df2){
  df <- df1 %>% 
    inner_join(df2,by="id")
  return(df)
}

comb = prepare_data(dataScored, dataROI)
comb$Quality = as.factor(comb$Quality)
comb$dist = as.factor(comb$dist)
comb$Biopsy = as.factor(comb$Biopsy)
comb$BI.RADS = as.factor(comb$BI.RADS)

tbl <- with(comb, table(Quality, dist))
addmargins(tbl)

barplot(tbl,xlab="dist", ylab="Quality",
        col=c("khaki", "cyan", "coral"),legend=rownames(tbl),beside=T)

result = chisq.test(comb$Quality, comb$dist); result; result$expected
result = chisq.test(comb$Biopsy, comb$dist); result; result$expected
result = chisq.test(comb$BI.RADS, comb$dist); result; result$expected
