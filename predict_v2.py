from transformers import pipeline
from sklearn.metrics import classification_report
import spacy
def preprocess_with_spacy(texts, language_model="zh_core_web_sm"):
    """
    使用 spaCy 對文本進行預處理。
    Args:
        texts (list): 原始文本列表。
        language_model (str): spaCy 語言模型名稱。
    Returns:
        list: 預處理後的文本列表。
    """
    # 加載 spaCy 語言模型
    nlp = spacy.load(language_model)

    # 預處理文本
    processed_texts = []
    for text in texts:
        doc = nlp(text)
        # 僅保留有意義的詞（去除停用詞和標點符號）
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        processed_texts.append(" ".join(tokens))
    
    return processed_texts

def predict_with_model(model_path, texts):
    """
    使用微調後的模型進行關鍵字相關性預測。
    Args:
        model_path (str): 微調後的模型路徑。
        texts (list): 待分析的文本列表。
    Returns:
        list: 每條文本的預測結果。
    """
    # 加載微調後的模型和分詞器
    classifier = pipeline("text-classification", model=model_path)

    # 預測文本
    results = classifier(texts)
    predicted_labels = [1 if result["label"] == "LABEL_1" else 0 for result in results]
    return predicted_labels

if __name__ == "__main__":
    model_path = "./fine_tuned_model"  # 微調後的模型路徑
    texts = [
        "這篇文章討論了最新的資訊安全漏洞防火牆。",
        "今天的天氣真不錯。",
        "社交工程是一個很好用的技術。",
        "這段文本是資料庫的敘述。",
        "入侵檢測的設置是資訊安全的重要環節。",
        "未經授權的訪問可能會導致數據洩漏。",
        "昨天我去看了一場電影。",
        "代理伺服器攻擊的手段越來越高明。",
        "加密技術是保障數據傳輸安全的關鍵。",
        "阿阿阿this is text",
        "nonono",
        "這是一個沒有任何關鍵字的句子。",
        "text text text",
        "我是失敗品",
        "這個專題好難哦",
        "為甚麼訓練都會掛掉",
        "語意分析努力中...",
        "oh my god you are the master",
        "XSS跨網站指令碼"
    ]

    # 真實標籤（假設的標籤，用於計算性能指標）
    true_labels = [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 1 表示相關，0 表示不相關

    # 使用 spaCy 進行文本預處理
    processed_texts = preprocess_with_spacy(texts, language_model="zh_core_web_sm")

    # 預測
    predicted_labels = predict_with_model(model_path, processed_texts)

    # 顯示每條文本的預測結果
    for text, processed_text, prediction, true_label in zip(texts, processed_texts, predicted_labels, true_labels):
        print(f"原始文本: {text}")
        print(f"預處理後文本: {processed_text}")
        print(f"預測標籤: {prediction}, 真實標籤: {true_label}")

    # 計算並顯示分類報告
    print("\n分類報告:")
    print(classification_report(true_labels, predicted_labels, target_names=["不相關", "相關"]))
