
# 台股高低點區間預測網站

## 功能
- 預測未來 3～30 個交易日內可能的最低價與最高價
- 技術指標：RSI、MACD、KD、布林通道
- 成交量特徵
- 台灣加權指數 (^TWII) 市場特徵
- 5 日相對強弱
- 時間序列交叉驗證
- 歷史 MAE
- 經驗式 80% 不確定區間
- Streamlit 網頁介面

## 本機執行
```bash
pip install -r requirements.txt
streamlit run app.py
```

執行後瀏覽器通常會開啟：
http://localhost:8501

## 部署成公開網站
最簡單方式是 Streamlit Community Cloud：

1. 建立 GitHub repository。
2. 上傳 app.py 與 requirements.txt。
3. 到 Streamlit Community Cloud 登入。
4. 選擇你的 GitHub repository。
5. Main file path 設為 app.py。
6. Deploy。
7. 部署完成後會得到公開網址，例如：
   https://你的專案名稱.streamlit.app

## 注意
本工具是統計 / 機器學習估計，不可能保證真正的最高點或最低點，也不構成投資建議。
