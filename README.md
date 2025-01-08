文章檢索小助手

使用教學：
1.使用 pdf_to_txt.py 將中英文論文轉換成txt格式
2.調整 ollama.py 內容路徑與檔案名稱並執行 

補充：
1.使用前須要先訓練模型(train_v4.py)，因為模型檔案太大沒辦法直接放上來，我的模型準確率約74%。
2.訓練模型會使用到keyword.txt，可以自行調整訓練內容，我們採用遷移學習的方式訓練，需要先使用(txt_to_csv_keyword.py)將keyword.txt與not_keyword.txt轉換成keyword_dataset.csv並帶入train_v4.py訓練。
3.可以使用 predict_v2 對模型進行預測。
