import csv

def prepare_dataset_from_keywords(keyword_file, output_csv, unrelated_keyword_file=None):
    """
    使用相關和不相關關鍵字生成數據集，方便後續模型訓練。
    Args:
        keyword_file (str): 包含相關關鍵字的 .txt 文件。
        output_csv (str): 生成的數據集 .csv 文件路徑。
        unrelated_keyword_file (str, optional): 包含不相關關鍵字的 .txt 文件。如果提供，將標記為 0。
    """
    with open(keyword_file, 'r', encoding='utf-8') as kf:
        keywords = [line.strip() for line in kf.readlines() if line.strip()]

    unrelated_keywords = []
    if unrelated_keyword_file:
        with open(unrelated_keyword_file, 'r', encoding='utf-8') as ukf:
            unrelated_keywords = [line.strip() for line in ukf.readlines() if line.strip()]

    # 使用 utf-8-sig 編碼保存 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text", "label"])  # 定義列名
        
        # 添加相關關鍵字
        for keyword in keywords:
            writer.writerow([keyword, 1])  # 1 表示相關

        # 添加不相關關鍵字
        for keyword in unrelated_keywords:
            writer.writerow([keyword, 0])  # 0 表示不相關

    print(f"數據集已保存至：{output_csv}")

if __name__ == "__main__":
    keyword_file = "keywords.txt"  # 相關關鍵字文件
    unrelated_keyword_file = "not_keywords.txt"  # 不相關關鍵字文件（可選）
    output_csv = "keyword_dataset.csv"  # 數據集文件路徑

    # 生成基於關鍵字的數據集
    prepare_dataset_from_keywords(keyword_file, output_csv, unrelated_keyword_file)
