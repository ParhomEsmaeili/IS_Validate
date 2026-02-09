#!/bin/bash
declare -ax MASTER_APPS=("nnintv1" "adadesign1" "adadesign2")

export MASTER_SPLIT="designset"
source ./result_summarisation/results_aggregation_run.sh
source ./result_summarisation/standard_metric_summary_run.sh
source ./oracle_metrics/num_of_interactions_run.sh
source ./ranking_and_stat_tests/execute_sig_and_ranking.sh
source ./table_generation/table_generator_run.sh
