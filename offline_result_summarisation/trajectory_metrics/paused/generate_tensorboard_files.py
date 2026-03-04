#TODO: Amend this after the pseudo-time AUC results have been regenerated, structure should not change much, but
# the logic of how we put together the pseudotime metrics will.

# import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
# from pathlib import Path

# def generate_tensorboard_files(df: pd.DataFrame, output_dir: str = "./runs") -> None:
#     """
#     Generate TensorBoard files from a pseudotime-indexed dataframe.
    
#     Args:
#         df: DataFrame with pseudotime index and columns 'Dice_auc_scores' and 'NSD_auc_scores'
#         output_dir: Directory to save TensorBoard logs
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     writer = SummaryWriter(log_dir=output_path)
    
#     # Iterate through dataframe with pseudotime as step
#     for pseudotime, row in df.iterrows():
#         if "Dice_auc_scores" in df.columns:
#             writer.add_scalar(
#                 "Metrics/Dice_auc_scores",
#                 row["Dice_auc_scores"],
#                 global_step=int(pseudotime)
#             )
        
#         if "NSD_auc_scores" in df.columns:
#             writer.add_scalar(
#                 "Metrics/NSD_auc_scores",
#                 row["NSD_auc_scores"],
#                 global_step=int(pseudotime)
#             )
    
#     writer.close()
#     print(f"TensorBoard files generated in {output_path}")


# if __name__ == "__main__":
#     # Example usage
#     df = pd.read_csv("your_data.csv", index_col="pseudotime")
#     generate_tensorboard_files(df)