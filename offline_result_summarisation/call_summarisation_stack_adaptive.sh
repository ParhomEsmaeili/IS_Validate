#!/bin/bash
export MASTER_PRETRAINED="nnintv1" #This is used to determine which pretrained method to pull results from for
#padding purposes on the pseudo-time conversion!
export MASTER_SPLIT="holdoutset" #"designset"
declare -ax MASTER_DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
declare -ax MASTER_DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010")
# declare -ax MASTER_DATASET_NAMES=("Dataset001_BrainTumour" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset010_Colon")
# declare -ax MASTER_DATASET_IDS=("001" "004" "005" "006" "007" "010")

# CONF_NAME="adatest2_nnint_segvol_comparison_medians"

# CONF_NAME="adatest2_adatest5_adatest6_nnint_comparison_medians"
# declare -ax MASTER_NNUNET_STATISTIC=("quantile") #("gaussian")
# declare -ax MASTER_NNUNET_BOUND=("0.5") #("0") #("0.5") #("0")
# declare -ax MASTER_DICE_AUC_STATISTIC=("mean") #The statistic which we report/determine betterness with.
# declare -ax MASTER_NSD_AUC_STATISTIC=("mean")  #The statistic which we report/determine betterness with.
# declare -ax MASTER_DICE_ITERATION_STATISTIC=("median") ##The statistic which we report/determine betterness with.
# declare -ax MASTER_NSD_ITERATION_STATISTIC=("median") #The statistic which we report/determine betterness with.

CONF_NAME="adatest2_adatest5_nnint_comparison_means" 
declare -ax MASTER_NNUNET_STATISTIC=("gaussian") #("quantile") #("gaussian")
declare -ax MASTER_NNUNET_BOUND=("0") #("0.5") #("0") #("0.5") #("0")
declare -ax MASTER_DICE_AUC_STATISTIC=("mean") #The statistic which we report/determine betterness with.
declare -ax MASTER_NSD_AUC_STATISTIC=("mean")  #The statistic which we report/determine betterness with.
declare -ax MASTER_DICE_ITERATION_STATISTIC=("mean") ##The statistic which we report/determine betterness with.
declare -ax MASTER_NSD_ITERATION_STATISTIC=("mean") #The statistic which we report/determine betterness with.

declare -ax MASTER_NOI_STATISTIC=("mean") #The statistic which we report/determine betterness with.
# declare -ax MASTER_FAILURE_FRACTION_STATISTIC= Not required, as fraction is a single number.
declare -ax MASTER_PSEUDOTIME_NOS_EPOCH=("Interactive Edit Iter 100") #The epoch at which we want to extract the number of samples metric, this should ideally be the same epoch at which we extract the performance metric for the pseudotime AUC calculation, as this is the most relevant statistic to report in the paper.

# # # # declare -ax MASTER_APPS=("sammed2dv1" "sam2v1" "segvolv1" "sammed3dv1" "nnintv1")
# # # # declare -ax MASTER_APPS=("segvolv1" "nnintv1") 
# declare -ax MASTER_APPS=("nnintv1")
# # # # # #First we run aggregation and result extraction on just the pretrained method.
# declare -ax MASTER_RUN_NUMS=("1" "2" "3")  #("1" "1" "1") #("1" "2" "3")
# source ./result_summarisation/results_aggregation_run.sh
# # # # # # Now lets extracted expected performances on the holdout set, per-run and aggregated, for the pretrained method.
# declare -ax MASTER_RUN_NUMS=("1" "2" "3" "-aggregated") #("1" "1" "1" "-aggregated") #("1" "2" "3" "-aggregated") 
# source ./result_summarisation/standard_metric_summary_run.sh
# # # # # #Now calculate the NoI metrics for the pretrained method.
# source ./oracle_metrics/num_of_interactions_run.sh

# # # # # # # # #Now lets generate per-run metrics for the adaptive methods. 
# declare -ax MASTER_APPS=("adatest2" "adatest5") # "adatest6") #("adatest2")
# declare -ax MASTER_RUN_NUMS=("1" "2" "3")
# # # # # # # #First lets calculate the summarisation metrics and NoI metrics on the original runs. 
# # # # # # # #We will need to do this for all episodes, logic must be handled within the bash script.
# source ./result_summarisation/standard_metric_summary_run.sh
# source ./oracle_metrics/num_of_interactions_run.sh



# # # # # # # # # # # #Now we have the metrics for the original runs for all episodes. 
# # # # # # # # # # # #Lets also do the same on averaged runs on a per-episode basis (the same set of test cases are being used!)
# # # # # # # # # # # We will also calculate the same on averaged run of the FINAL episode from each run. 
# # # # # # # # # # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # # # # # # # # # # NOTE: For now, with fixed length episodes the final episode aggregate should be the same as the aggregate over
# # # # # # # # # # # last episode across all runs. BUT, if the episode length wasn't fixed then this would not be the case. Ultimately
# # # # # # # # # # # the most relevant statistic here when reporting a single table is the aggregate of FINAL episode across runs. We
# # # # # # # # # # # only calculate the aggregate across all episodes for sanity checking for now/while we use a fixed length episode procedure.
# # # # # # # # # # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # # # # # # # # # # The final episode should ideally be the one that has seen the most data/i.e most comparable to nnU-Net in terms of
# # # # # # # # # # # data availability. Though this is also part of the meta-algorithm design.
# # # # # # # # # # # 


# source ./result_summarisation/results_aggregation_run.sh
# source ./result_summarisation/results_aggregation_run_final_episode.sh 
# declare -ax MASTER_RUN_NUMS=("-aggregated") #We run it only on the aggregated runs now.
# source ./result_summarisation/standard_metric_summary_run.sh
# source ./oracle_metrics/num_of_interactions_run.sh

# declare -ax MASTER_FINAL_EPISODE_ONLY_FLAG=true
# source ./result_summarisation/standard_metric_summary_run.sh #We run the final episode summarisation separately.
# source ./oracle_metrics/num_of_interactions_run.sh #We run the NoI metrics on the final episode separately as well, because this is the most relevant statistic to report in the paper, and also because it can be volatile across episodes so we want to make sure we are reporting the most stable/relevant statistic for this metric.
# # # # Now that we are done with that... lets clear the FINAL_EPISODE_ONLY_FLAG for downstream analyses.
# unset MASTER_FINAL_EPISODE_ONLY_FLAG

# # # # # # # # # #Now we must convert the episodic performance into a pseudotime metric.

# # # # # # # # Now lets convert the original runs into pseudotime trajectories, we now append nnintv1. this is the baseline.
# declare -ax MASTER_APPS=("nnintv1" "adatest2" "adatest5") # "adatest6") #"adatest2" "adatest4" "adatest5" "adatest6") #"adatest2")
# declare -ax MASTER_RUN_NUMS=("1" "2" "3")
# source ./trajectory_metrics/merge_episodes_run.sh 
# # # # # # # # # #Now we aggregate the pseudotime metrics across the original runs, to get a single pseudotime trajectory
# # # # # # # # # #for each relevant metric. 

# # # # # # # # # # # # # #Now lets aggregate the pseudotime metrics.
# declare -ax MASTER_RUN_NUMS=("1" "2" "3") #("1" "1" "1") #("1" "2" "3")
# source ./trajectory_metrics/pseudotime_results_aggregation_run.sh

# # # # # # # # # # #We now declare the original run nums and master run num separately, we want to obtain bounds.
# declare -ax MASTER_ORIGINAL_RUN_NUMS=("1" "2" "3") #("1" "1" "1") #("1" "2" "3")
# declare -ax MASTER_RUN_NUMS=("-aggregated")
# # # # # #Now, calculate the pseudotime AUCs for each relevant metric desired
# source ./trajectory_metrics/pseudotime_auc_run.sh

# # # # # # # # # # # #Nowe we generate the numbner of samples metric.
# # # # # # # declare -ax MASTER_APPS=("sammed2dv1" "sam2v1" "segvolv1" "sammed3dv1" "nnintv1") 
# declare -ax MASTER_APPS=("nnintv1" "adatest2" "adatest5") # "adatest6") #"adatest4" "adatest5" "adatest6") #"adatest3")
# declare -ax MASTER_RUN_NUMS=("-aggregated") #We run it only on the aggregated runs now, as the single runs can be too volatile to obtain a meaningful metric here.
# source ./trajectory_metrics/num_of_samples_run.sh

# # # # # # # # # #Number crunching mostly over. Time for plotting and stat sig tests. this depends on the apps we have calculated, so
# # # # # # # # # #we keep it at the end here so we can avoid re-running the whole pipeline if we want to permute the combination of apps.

# # # # # # # #Now plot the pseudotime for each metric with the AUC vals indicated on the plot.
declare -ax MASTER_RUN_NUMS=("-aggregated") #We should only be plotting the

declare -ax MASTER_APPS=("nnintv1" "adatest2" "adatest5") # "adatest6") #"adatest4" "adatest5" "adatest6") #"adatest2")
declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_traj"
source ./trajectory_metrics/plot_pseudotime_auc_run.sh
# # # # # # # # source ./trajectory_metrics/generate_pseudotime_tensorboard.sh
# # 
source ./trajectory_metrics/plot_pseudotime_auc_selection_run.sh

declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_final_episode_comparison"
declare -ax MASTER_APPS=("nnintv1" "adatest2-episodefinal" "adatest5-episodefinal")
# # # declare -ax MASTER_APPS=("segvolv1" "nnintv1" "adatest2-episodefinal") #"adatest4-episodefinal" "adatest5-episodefinal" "adatest6-episodefinal") #"adatest2-episode3")
source ./ranking_and_stat_tests/execute_sig_and_ranking.sh

# # declare -ax MASTER_CONFIG_NAME="zero_shot_baseline_subset" #full"
# # declare -ax MASTER_APPS=("sammed2dv1" "sam2v1" "segvolv1" "sammed3dv1" "nnintv1")
# # declare -ax MASTER_APPS=("segvolv1" "nnintv1")
# # source ./ranking_and_stat_tests/execute_sig_and_ranking.sh


declare -ax MASTER_APPS=("nnintv1" "adatest2" "adatest5") # "adatest6") #"adatest4" "adatest5" "adatest6") #"adatest2")
declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_traj"
source ./ranking_and_stat_tests/execute_sig_and_ranking_traj.sh

# # #Now generate tables.
# # #First the snapshot.

# # declare -ax MASTER_APPS=("sammed2dv1" "sam2v1" "segvolv1" "sammed3dv1" "nnintv1")
# # declare -ax MASTER_APPS=("segvolv1" "nnintv1")
# # declare -ax MASTER_CONFIG_NAME="zero_shot_baseline_subset"
# # source ./table_generation/table_generator_run.sh

declare -ax MASTER_APPS=("nnintv1" "adatest2-episodefinal" "adatest5-episodefinal") # "adatest6-episodefinal") #"adatest4-episodefinal" "adatest5-episodefinal" "adatest6-episodefinal") #"adatest2-episode3")
declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_final_episode_comparison"
source ./table_generation/table_generator_run.sh

# #Then the trajectory table.
declare -ax MASTER_APPS=("nnintv1" "adatest2" "adatest5") # "adatest6") #"adatest4" "adatest5" "adatest6") #"adatest2")
declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_traj"
source ./table_generation/table_generator_run_traj.sh