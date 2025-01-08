import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_path):
    """
    加載已訓練的模型和分詞器。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def classify_paragraphs(paragraphs, tokenizer, model):
    """
    使用模型對段落進行分類，返回資安相關的段落。
    """
    security_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probabilities).item()
        if label == 1:  # 1 表示與資安相關
            security_paragraphs.append((i + 1, paragraph))  # 保存段落編號與內容
    return security_paragraphs


def split_text_into_paragraphs(text):
    """
    將輸入的文本按段落分割。
    """
    return [p.strip() for p in text.split("\n") if p.strip()]


def load_text_from_file(file_path):
    """
    從文本檔案中讀取內容。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"檔案未找到：{file_path}")
        return None


def load_keywords_from_file(file_path):
    """
    從關鍵字檔案中讀取關鍵字列表。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"關鍵字檔案未找到：{file_path}")
        return []


def find_keywords_in_paragraphs(paragraphs, keywords):
    """
    對段落進行關鍵字比對，返回包含關鍵字的段落及編號。
    """
    keyword_matches = []
    for idx, paragraph in paragraphs:
        for keyword in keywords:
            if keyword in paragraph:
                keyword_matches.append((idx, keyword, paragraph))
                break  # 段落中找到一個關鍵字即停止繼續檢查
    return keyword_matches


def main():
    model_path = "./fine_tuned_model"  # 您的模型路徑
    text_file_path = "植基於邊緣偵測及最佳像素調整之資訊隱藏方法.txt"  # 載入的文本檔案名稱
    keywords_file_path = "keywords.txt"  # 關鍵字檔案名稱

    # 加載模型和分詞器
    tokenizer, model = load_model(model_path)

    # 載入文本檔案
    text = load_text_from_file(text_file_path)
    if text is None:
        return

    # 載入關鍵字檔案
    keywords = load_keywords_from_file(keywords_file_path)
    if not keywords:
        print("系統: 關鍵字列表為空，請檢查 keywords.txt 檔案。")
        return

    # 分割段落
    paragraphs = split_text_into_paragraphs(text)

    # 使用模型分類段落
    security_paragraphs = classify_paragraphs(paragraphs, tokenizer, model)

    # 對資安相關段落進行關鍵字比對
    keyword_matches = find_keywords_in_paragraphs(security_paragraphs, keywords)

    print(f"系統: 文本已處理，共有 {len(paragraphs)} 段落，其中 {len(security_paragraphs)} 段與資安相關。")
    print("輸入指令開始操作，'help' 顯示可用指令，'exit' 結束程序。")

    range_start = None
    range_end = None

    while True:
        command = input("\n輸入指令: ").strip()
        if command.lower() == "exit":
            print("系統: 再見！")
            break
        elif command.lower() == "help":
            print("""
可用指令：
- ls: 列出所有段落編號
- ls 1: 列出指定段落的內容，並設定當前範圍
- ls cyber: 列出資安相關且匹配關鍵字的段落編號
- up x: 擴展當前範圍的前 x 段，並顯示合併內容
- down x: 擴展當前範圍的後 x 段，並顯示合併內容
- exit: 結束程序
            """)
        elif command.lower() == "ls":
            print("\n系統: 所有段落編號：")
            for i in range(len(paragraphs)):
                print(f"段落 {i + 1}")
        elif command.lower() == "ls cyber":
            print("\n系統: 資安相關且匹配關鍵字的段落：")
            for idx, keyword, paragraph in keyword_matches:
                print(f"段落 {idx}（匹配關鍵字: {keyword}）")
        elif command.lower().startswith("ls "):
            try:
                idx = int(command.split()[1]) - 1
                if 0 <= idx < len(paragraphs):
                    range_start = idx
                    range_end = idx
                    print(f"\n段落 {idx + 1}：\n{paragraphs[idx]}")
                else:
                    print("系統: 段落編號超出範圍。")
            except (ValueError, IndexError):
                print("系統: 請輸入有效的段落編號，例如 'ls 3'")
        elif command.lower().startswith("up "):
            try:
                count = int(command.split()[1])
                if range_start is None or range_end is None:
                    print("系統: 請先使用 'ls 1' 選擇一個段落。")
                else:
                    range_start = max(0, range_start - count)
                    combined_content = ''.join(paragraphs[range_start:range_end + 1])
                    print(f"\n段落 {range_start + 1}~{range_end + 1}：\n{combined_content}\n")
            except ValueError:
                print("系統: 請輸入有效的數字，例如 'up 2'")
        elif command.lower().startswith("down "):
            try:
                count = int(command.split()[1])
                if range_start is None or range_end is None:
                    print("系統: 請先使用 'ls 1' 選擇一個段落。")
                else:
                    range_end = min(len(paragraphs) - 1, range_end + count)
                    combined_content = ''.join(paragraphs[range_start:range_end + 1])
                    print(f"\n段落 {range_start + 1}~{range_end + 1}：\n{combined_content}\n")
            except ValueError:
                print("系統: 請輸入有效的數字，例如 'down 2'")
        else:
            print("系統: 無效的指令，輸入 'help' 查看可用指令。")


if __name__ == "__main__":
    main()
