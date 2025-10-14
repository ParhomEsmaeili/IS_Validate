
#This is an example of how one might write a bash script for running an evaluation with the current formulation of the codebase.
python3 run.py \
	--data_root="root_folder_containing_datasets_dir" \
	--app_root="root_folder_containing_input_applications_dir" \
	--metrics_root="root_folder_for_where_each_experiments_metrics_are_stored" \\
	--seg_root="root_folder_for_where_each_experiments_segs_are_stored" 

