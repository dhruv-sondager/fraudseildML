import pandas as pd
from pathlib import Path

input_folder = Path("data")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

for file in input_folder.glob("*.xlsx"):
    print("Converting:", file.name)

    df = pd.read_excel(file)

    output_file = output_folder / (file.stem + ".parquet")
    df.to_parquet(
        output_file,
        compression="zstd",
        index=False
    )

    print("Saved:", output_file)
    print("Size:", round(output_file.stat().st_size / 1024**2, 2), "MB")