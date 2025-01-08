from pdfminer.high_level import extract_text
import os
import re

def clean_repeated_characters(text):
    """
    清理文本中重複的字元或詞組，例如：
    '植植基基於於' -> '植基於'
    """
    try:
        # 處理單字重複，例如 "植植基基" -> "植基"
        text = re.sub(r'([\u4e00-\u9fa5])\1', r'\1', text)
        # 處理多字符重複，例如 "基基於於" -> "基於"
        text = re.sub(r'(([\u4e00-\u9fa5])\2)+', r'\2', text)
        return text
    except Exception as e:
        print(f"Error during character cleaning: {e}")
        return text

def clean_text_spaces(text):
    """
    清理文本中的多餘空格和段內多餘換行。
    """
    try:
         # 移除括号内的内容，包括中英文括号
        text = re.sub(r'\([^()]{1,5}\)', '', text)  # 移除长度在1到20的括号内容
        text = re.sub(r'（[^（）]{1,5}）', '', text)  # 处理中文括号
        # 合併多餘的空格
        text = re.sub(r'\s+', ' ', text)
        # 移除段內多餘換行符，僅保留段與段之間的單一換行
        text = re.sub(r'(?<!\n)\n(?!\n)', '', text)
        # 確保段與段之間保留單一換行
        text = re.sub(r'\n{2,}', '\n', text)
        
        return text.strip()
    except Exception as e:
        print(f"Error during space cleaning: {e}")
        return text
    
def remove_garbage_characters(text):
    """
    移除文本中的亂碼或非預期字符。
    """
    try:
        # 僅保留中文、英文、數字和常見標點符號
        text = re.sub(r'[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a\u0030-\u0039\u0020-\u007e\n，。！？“”‘’（）【】《》；：、]', '', text)
        return text
    except Exception as e:
        print(f"Error during garbage removal: {e}")
        return text
    
def pdf_to_txt(pdf_path, txt_path):
    """
    提取 PDF 文本，清理重複字元和多餘空格，保存為 TXT 文件。
    """
    try:
        # 提取 PDF 文本
        print(f"正在提取 PDF 文本：{pdf_path}")
        text = extract_text(pdf_path)

        if not text.strip():
            print(f"警告：PDF 文件中未提取到有效文本！")
            return

        # 清理重複內容
        text = clean_repeated_characters(text)

        # 清理空格和段內多餘換行
        clean_text = clean_text_spaces(text)
        clean_text = remove_garbage_characters(text)

        # 將文本保存為 TXT 文件
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(clean_text)
        print(f"文本成功提取並保存至：{txt_path}")
    except Exception as e:
        print(f"出錯了：{e}")

if __name__ == "__main__":
    # PDF 文件路徑
    pdf_path = "植基於邊緣偵測及最佳像素調整之資訊隱藏方法.pdf"

    # 保存的 TXT 文件路徑
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"  # 將 PDF 文件名更改為 .txt

    # 調用函數提取文本並保存
    pdf_to_txt(pdf_path, txt_path)
