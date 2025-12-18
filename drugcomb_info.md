## DrugComb Dataset Size for Drug Synergy Prediction

Based on my web search, here are the comprehensive statistics for the DrugComb dataset:

### Overall DrugComb Database (2021 Update)

As of the March 2021 update (published in Nucleic Acids Research, July 2021):

* **Drug combinations:** 751,498
* **Single drug screenings:** 717,684
* **Total unique data points:** 21,621,279
* **Cell lines:** 2,320 (from 225 cancer types + 3 infectious diseases)
* **Drugs:** 8,397 unique drugs
* **Studies curated:** 37 studies

### Benchmark Datasets Used in Research

For machine learning research on drug synergy prediction, researchers typically use processed subsets:

* **Balanced benchmark dataset:** 25,758 unique drug-drug-cell line combinations (integrated from GDSC and DrugComb)
* **NCI-ALMANAC subset:** 297,098 unique drug combination-cell line pairs (129 drugs tested across 59 cell lines)

### Your Specific Dataset

The `summary_v_1_5.csv` file in your codebase appears to be a curated version from the DrugComb portal. Based on your code:

* Features per sample: **8,551 dimensions** (drug signatures + cell line gene expression)
* The dataset undergoes filtering to remove invalid synergy scores and missing values
* Uses data augmentation (each sample provided twice with reversed drug order)

### Historical Context

The original DrugComb (2019) contained 437,923 drug combination experiments, which was substantially expanded to 751,498+ combinations in the 2021 update.

---

**Sources:**

* [DrugComb update: a more comprehensive drug sensitivity data repository and analysis portal - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218202/)
* [DrugComb update - Nucleic Acids Research](https://academic.oup.com/nar/article/49/W1/W174/6290546)
* [Drug Synergy Prediction - TDC](https://tdcommons.ai/multi_pred_tasks/drugsyn/)
* [Augmented drug combination dataset - Nature Scientific Reports](https://www.nature.com/articles/s41598-024-51940-9)
* [DDoS: a graph neural network based drug synergy prediction algorithm](https://arxiv.org/html/2210.00802v3)
