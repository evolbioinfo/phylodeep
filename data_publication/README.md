# PhyloDeep Data

This folder contains data used for the Figures and Tables in the PhyloDeep manuscript:

## Preprint

Voznica J, Zhukova A, Boskova V, Saulnier E, Lemoine F, Moslonka-Lefebvre M, Gascuel O (2021)
__Deep learning from phylogenies to uncover the transmission dynamics of epidemics__. [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.11.435006v1)

## Figures

### Fig_3 : 
   - you will find the test trees (100 per BD, BDEI and BDSS) in newick format in the folder [Fig_3/test_trees](Fig_3/test_trees)
   - you will find the predicted values (100 per model (BD, BDEI and BDSS) and per method (BEAST2, FFNN-SS, CNN-CBLV together with target values)) in the folder [Fig_3/predicted_values](Fig_3/predicted_values)
   - you will find all the BEAST2 logs (100 per model) in the folder [Fig_3/BEAST2_log](Fig_3/BEAST2_log)

### Fig_4 : 
   - Fig4 has its own [README](Fig_4/README.md)

### Fig_5 : 
  - you will find the BEAST2 logs for HIV phylogeny for each model in the folder [Fig_5/BEAST2_logs](Fig_5/BEAST2_logs)
  - you will find the values inferred with FFNN-SS and CNN-CBLV together with CI values in the folder [Fig_5/inferred_values](Fig_5/inferred_values)

### Supp_Fig_2 : 
   - you will find the test trees (100 per model (BD, BDEI)) in newick format in the folder [Supp_Fig_2/test_trees](Supp_Fig_2/test_trees)
   - you will find the predicted values (100/10,000 per model (BD, BDEI) and per method (BEAST2, FFNN-SS, CNN-CBLV together with target values)) in the folder [Supp_Fig_2/predicted_values](Supp_Fig_2/predicted_values)
   - you will find all the BEAST2 logs (100 per model (BD, BDEI)) in the folder [Supp_Fig_2/BEAST2_logs](Supp_Fig_2/BEAST2_logs)

### Supp_Fig_3 :
  - you will find predicted and target values (1,000 sets per network (FFNN-SS, CNN-CBLV) and per model (BD, BDEI and BDSS)) in the folder [Supp_Fig_3](Supp_Fig_3)

### Supp_Fig_4 :
  - Supp_Fig_4 has its own [README](Supp_Fig_4/README.md)

### Supp_Fig_5 :
  - you will find predicted and target values for the BDSS model (10,000 sets per network (CNN-CBLV, CNN-CRV and FFNN-CBLV) and per training set size setting (10K, 100K, 1M and 4M) 
in the folder [Supp_Fig_5/BDSS_large](Supp_Fig_5/BDSS_large)

### Supp_Fig_6 :
  - you will find predicted and target values (10,000 per model (BD, BDEI, BDSS) and method (FFNN-SS, CNN-CBLV, FFNN-SS-NULL and linear regression) 
with different training set size (10K, 100K, 1M and 4M)) together with predicted values by BEAST2 (100 sets) in the folder [Supp_Fig_6/predicted_values](Supp_Fig_6/predicted_values)


### Supp_Fig_7 :
  - you will find predicted values for the BDSS model (10,000 sets per network (CNN-CBLV, FFNN-SS and FFNN-origSS) in the folder [Supp_Fig_7](Supp_Fig_7)

## Tables

### Supp_Table_1_2 : 
   - you will find the test trees (10,000 per model (BD, BDEI, BDSS) and per tree size (small and big trees)) in newick format in the folder [Supp_Table_1_2/test_trees](Supp_Table_1_2/test_trees)
   - you will find the predicted values (100/10,000 per model (BD, BDEI, BDSS) and per method (BEAST2, FFNN-SS, CNN-CBLV, linear regression, FFNN-CBLV, FFNN-SS-NULL together with target values)) 
in the folder [Supp_Table_1_2/predicted_values](Supp_Table_1_2/test_trees)

### Supp_Table_5 : 
  - you will find the results of tree selection (10,000 per tree size (small or big trees) and per network (FFNN-SS and CNN-CBLV)) 
in folders [Supp_Table_5/FFNN_SS](Supp_Table_5/FFNN_SS) and [Supp_Table_5/CNN_CBLV](Supp_Table_5/CNN_CBLV) respectively
  - you will find corresponding BEAST2 logs (100 per tree size (small or big trees), per target model (BD, BDEI and BDSS) and per assessed model (BD, BDEI and BDSS)) 
in the folder [Supp_Table_5/BEAST2](Supp_Table_5/BEAST2). To avoid duplicates, if the target model (model of simulations) and assessed model were the same, you can find the BEAST2 logs in the folder [Fig_3](Fig_3)

### Supp_Table_7 : 
   - you will find the CI values (100 sets per model (BD, BDEI, BDSS), per tree size (small and big trees) and per method (BEAST2, FFNN-SS and CNN-CBLV)) in the folder [Supp_Table_7](Supp_Table_7)


