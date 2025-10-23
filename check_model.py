# check_model.py (偵錯版本)
import google.generativeai as genai
import os
from dotenv import load_dotenv

print("--- 程式開始 ---")

print("--- 1. 正在載入 .env 檔案... ---")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("!!! 錯誤：在 .env 檔案中找不到 'GEMINI_API_KEY'。")
    print("--- 程式結束 ---")
    exit()

# 為了安全，只顯示金鑰的前4碼和後4碼
print(f"--- 2. 成功載入金鑰: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]} ---")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("--- 3. 成功設定金鑰 (genai.configure) ---")

    print("--- 4. 正在呼叫 genai.list_models()... (請稍候) ---")
    models_list = list(genai.list_models()) # 轉換為列表
    print(f"--- 5. API 回傳了 {len(models_list)} 個模型 ---")

    if len(models_list) == 0:
        print("!!! 錯誤：您的 API 金鑰回傳了 0 個可用模型。")
        print("!!! 請嘗試重新整理 aistudio.google.com 頁面並「建立新金鑰」。")
    
    found_one = False
    for m in models_list:
        # 我們只關心 'generateContent' (生成內容) 的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"    -> 找到可用模型: {m.name}")
            found_one = True

    if not found_one and len(models_list) > 0:
        print("--- 警告：找到了模型，但沒有一個支援 'generateContent' ---")
    
    print("--- 6. 查詢完畢 ---")

except Exception as e:
    print(f"!!! 查詢時發生嚴重錯誤: {e}")
    print("--- 程式結束 ---")