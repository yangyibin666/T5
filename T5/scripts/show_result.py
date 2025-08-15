import pandas as pd


def csv_to_markdown_table(csv_path, n=10):
    df = pd.read_csv(csv_path)
    # 只显示前 n 行，防止内容过多
    print(df.head(n).to_markdown(index=False))


if __name__ == "__main__":
    csv_to_markdown_table('results/summary_compare.csv', n=10)
