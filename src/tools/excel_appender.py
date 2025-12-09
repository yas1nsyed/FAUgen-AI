import pandas as pd

def append_excels(output_path: str, *excel_paths: str):
    final_df = None

    for idx, file_path in enumerate(excel_paths):
        # Read header as data
        df = pd.read_excel(file_path, header=None)

        # Skip header row for 2nd+ files
        if idx > 0:
            df = df.iloc[1:]

        if final_df is None:
            final_df = df
        else:
            final_df = pd.concat([final_df, df], ignore_index=True)

    # Write output, restore header row as header
    final_df.columns = final_df.iloc[0]   # first row becomes header
    final_df = final_df[1:]               # drop that row

    final_df.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")


append_excels(
    "combined_output.xlsx",
    "/home/ubuntu/projects/FAUgen-AI/src/excel_processing/aces_metadata_output.xlsx",
    "/home/ubuntu/projects/FAUgen-AI/pdf_sections_hierarchical.xlsx" # final output file path, excel file to combine 1, excel file to combine 2, ....
)

