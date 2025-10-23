# server.py (API 版本 - 已移除本地部署)
import os
import uuid
import sqlite3
import threading
import requests # <-- 使用 requests 呼叫 API
import time
import base64 # <-- SVD v1 需要
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- ↓↓↓ 移除本地 AI 模型的 Import ↓↓↓ ---
# import torch
# from diffusers import ...
# --- ↑↑↑ --------------------------- ---

# 導入 AI 函式庫
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# (移除 OpenCV，除非您未來有其他用途)
# import cv2

# --- 1. 初始化設定 ---
load_dotenv()
app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# --- 2. 設定 API 金鑰 ---
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY") # <-- Stability 金鑰
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# (選擇您要使用的搜尋 API，這裡是台灣記憶庫範例)
TAIWAN_MEMORY_API_KEY = os.getenv("TAIWAN_MEMORY_API_KEY")
TAIWAN_MEMORY_SEARCH_URL = "https://data.culture.tw/api/v1/search"
# (或者 Google 搜尋的金鑰設定)
# GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
# GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
# GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


# --- ↓↓↓ 還原 Stability API 端點 ↓↓↓ ---
T2I_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
SVD_API_URL = "https://api.stability.ai/v1/generation/stable-video-diffusion" # <-- 使用 v1 SVD
# --- ↑↑↑ --------------------------- ---

# (移除 GPU 檢查)

# --- 3. 資料庫設定 (不變) ---
DB_FILE = "jobs.db"
def init_db():
    db = sqlite3.connect(DB_FILE)
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY, status TEXT NOT NULL, text_content TEXT,
        image_prompt_en TEXT, image_url TEXT, video_url TEXT
    )""")
    db.commit()
    db.close()

# --- 4. AI 服務函式 ---

# server.py

def generate_gemini_content(theme):
    """(LLM) 使用 Gemini 生成課程對話(中文) 和 圖片提示詞(英文) - 強化剖析"""
    print(f"Gemini: 正在生成 '{theme}' 的內容...")
    model = genai.GenerativeModel('models/gemini-2.5-pro') # 使用您可用的模型
    prompt = f"""
    請為懷舊治療課程生成關於「{theme}」的內容，你需要提供兩部分：
    1.  dialogue: 一段溫暖的中文開場白 (約 50-80 字)。
    2.  image_prompt: 一段適合AI繪圖的、描述性的**英文**提示詞 (English prompt)，用來生成一张符合上述中文情境的懷舊照片 (例如 "A nostalgic photo of a bustling temple square in Taiwan, 1960s style..."，請將英文提示詞限制在 77 個 token 以內，保持簡潔有力)。

    請嚴格使用以下格式回傳，不要有任何額外文字：
    DIALOGUE: [這裡放中文對話]
    IMAGE_PROMPT: [Here is the English image prompt]
    """
    
    dialogue = f"這是一段關於「{theme}」的備用對話。(Gemini 失敗)" # 先設定備用內容
    image_prompt = f"A nostalgic photograph about '{theme}' in Taiwan, 1960s style, warm colors."

    try:
        response = model.generate_content(
             prompt,
             request_options={"timeout": 20.0} # 加入超時
        )
        text = response.text
        
        # --- ↓↓↓ 使用 find() 進行更強健的剖析 ↓↓↓ ---
        print(f"Gemini Raw Response:\n---\n{text}\n---") # 印出原始回應，方便除錯

        dialogue_start_tag = "DIALOGUE:"
        prompt_start_tag = "IMAGE_PROMPT:"
        
        # 尋找標籤的位置
        dialogue_start_index = text.find(dialogue_start_tag)
        prompt_start_index = text.find(prompt_start_tag)

        # 提取 Dialogue
        if dialogue_start_index != -1:
            # 如果 IMAGE_PROMPT 標籤存在，則 dialogue 在兩個標籤之間
            if prompt_start_index != -1 and prompt_start_index > dialogue_start_index:
                dialogue = text[dialogue_start_index + len(dialogue_start_tag) : prompt_start_index].strip()
            # 如果 IMAGE_PROMPT 標籤不存在，則 dialogue 是 DIALOGUE: 後的所有內容
            else:
                 dialogue = text[dialogue_start_index + len(dialogue_start_tag) :].strip()

        # 提取 Image Prompt
        if prompt_start_index != -1:
            # image_prompt 是 IMAGE_PROMPT: 後的所有內容
            # (我們假設 DIALOGUE: 總是在前面，如果不是，這裡會抓錯，但機率低)
             image_prompt = text[prompt_start_index + len(prompt_start_tag) :].strip()
             
        # (可選) 再次清理可能的多餘引號或標記
        dialogue = dialogue.strip('[] \n')
        image_prompt = image_prompt.strip('[] \n')
        # --- ↑↑↑ 剖析結束 ↑↑↑ ---

        print(f"Gemini Parsed: (中文) {dialogue}")
        print(f"Gemini Parsed: (英文) {image_prompt}")
        
        # 確保即使剖析不完美，也有備用值
        if not dialogue: dialogue = f"關於「{theme}」的備用對話。(剖析失敗)"
        if not image_prompt: image_prompt = f"Nostalgic photo of '{theme}', Taiwan."
            
        return dialogue, image_prompt
        
    except google_exceptions.DeadlineExceeded:
         print("Gemini 錯誤: 呼叫 Google API 超時 (超過 20 秒)。")
         # 回傳備用內容
    except Exception as e:
        print(f"Gemini 錯誤或剖析失敗: {e}") # 現在這裡會印出更詳細的錯誤
        # 回傳備用內容

    # 如果 try 區塊中發生任何錯誤，會執行到這裡並回傳備用值
    return dialogue, image_prompt


def search_taiwan_memory(query, job_id):
    # ... (這個搜尋函式保持不變) ...
    # (或者您換成 search_google_wikimedia)
    if not TAIWAN_MEMORY_API_KEY: return None, None
    print(f"TaiwanMemory: 正在搜尋: '{query}'")
    try:
        # ... (搜尋與下載邏輯不變) ...
        headers = {"X-API-KEY": TAIWAN_MEMORY_API_KEY}
        params = {"q": query, "limit": 10}
        response = requests.get(TAIWAN_MEMORY_SEARCH_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['result'] and data['result']['records']:
             for record in data['result']['records']:
                 # ... (找到圖片 URL 並下載儲存的邏輯) ...
                 if 'image' in record and record['image']:
                      image_url = record['image']
                      if not image_url.startswith('http'): continue
                      print(f"TaiwanMemory: 找到照片! 正在從 {image_url} 下載...")
                      img_response = requests.get(image_url, timeout=10)
                      img_response.raise_for_status()
                      filename = f"{job_id}.png"
                      filepath = Path("public/images") / filename
                      filepath.parent.mkdir(parents=True, exist_ok=True)
                      with open(filepath, "wb") as f: f.write(img_response.content)
                      print(f"TaiwanMemory: 圖片已儲存於 {filepath}")
                      public_url = f"/images/{filename}"
                      return str(filepath), public_url
        print("TaiwanMemory: 找不到相符的「圖片」資料。")
        return None, None
    except Exception as e:
        print(f"TaiwanMemory API 錯誤: {e}")
        return None, None

# --- ↓↓↓ (關鍵還原) T2I 函式 (API 版本) ↓↓↓ ---
def generate_t2i_image(prompt, job_id):
    """(T2I - API) 使用 Stability AI API 生成靜態圖片"""
    print("T2I (API): 正在呼叫 API 生成圖片...")
    try:
        # 使用我們之前修正好的 (None, value) 格式 for multipart
        payload = {
            "prompt": (None, prompt), # 使用 Gemini 提供的英文 Prompt
            "output_format": (None, "png"),
            "aspect_ratio": (None, "16:9"),
            "model": (None, "sd3.5-medium") # 或者 sd3-medium (如果 API 支援)
        }

        response = requests.post(
            T2I_API_URL,
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "image/*" # API 需要 image/* 或 application/json
            },
            files=payload # 使用 files= 強制 multipart
        )

        if response.status_code != 200:
            raise Exception(f"T2I API 錯誤: {response.text}")

        # 儲存圖片 (直接儲存 response.content)
        filename = f"{job_id}.png"
        filepath = Path("public/images") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(response.content)

        image_url = f"/images/{filename}"
        print(f"T2I (API): 圖片儲存於 {image_url}")
        return image_url, str(filepath) # 仍然回傳 URL 和本機路徑

    except Exception as e:
        print(f"T2I (API) 錯誤: {e}")
        return None, None
    # (不再需要 finally: del pipe ... 了)


# --- ↓↓↓ (關鍵還原) SVD 函式 (API 版本) ↓↓↓ ---
def generate_svd_video(image_path, job_id):
    """(SVD - API v1) 使用 Stability AI API 生成動態影片"""
    print(f"SVD (API Job: {job_id}): 任務開始，正在呼叫 API...")
    try:
        # SVD v1 API 需要上傳圖片檔案
        with open(image_path, "rb") as f:
            files_payload = {'image': f}

            response = requests.post(
                SVD_API_URL, # v1 端點
                headers={
                    "authorization": f"Bearer {STABILITY_API_KEY}",
                    "accept": "application/json" # v1 期望 JSON 回應
                },
                files=files_payload
            )

        if response.status_code != 200:
            try:
                error_json = response.json()
                raise Exception(f"SVD API 錯誤 (JSON): {error_json}")
            except requests.exceptions.JSONDecodeError:
                raise Exception(f"SVD API 錯誤 (Raw): {response.text}")

        # 處理 v1 API 的 Base64 JSON 回應
        response_json = response.json()
        if 'artifacts' not in response_json or not response_json['artifacts']:
            raise Exception("SVD API 錯誤: 回應中找不到 'artifacts'")

        video_base64 = response_json['artifacts'][0]['video']
        video_data = base64.b64decode(video_base64)

        # 儲存影片
        filename = f"{job_id}.mp4"
        filepath = Path("public/videos") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(video_data)

        video_url = f"/videos/{filename}"
        print(f"SVD (API Job: {job_id}): 影片完成！儲存於 {video_url}")
        return video_url

    except Exception as e:
        print(f"SVD (API Job: {job_id}) 錯誤: {e}")
        return None
    # (不再需要 finally: del pipe ... 了)


# --- 5. 背景任務 (不變) ---
def run_svd_background_task(job_id, image_local_path):
    # ... (這個函式保持不變，它現在會呼叫上面的 API 版本) ...
    video_url = generate_svd_video(image_local_path, job_id)

    db = sqlite3.connect(DB_FILE)
    if video_url:
        db.execute(
            "UPDATE jobs SET status = 'completed', video_url = ? WHERE job_id = ?",
            (video_url, job_id)
        )
        print(f"DB (Job: {job_id}): 狀態更新為 completed")
    else:
        db.execute("UPDATE jobs SET status = 'failed' WHERE job_id = ?", (job_id,))
        print(f"DB (Job: {job_id}): 狀態更新為 failed")
    db.commit()
    db.close()

# --- 6. API 端點 (不變) ---
@app.route('/')
def index():
    # ... (不變) ...
    return app.send_static_file('index.html')

@app.route('/api/generate-content', methods=['POST'])
def generate_content():
    # ... (這個主邏輯完全不變) ...
    theme = request.json.get('theme', '廟口')
    job_id = str(uuid.uuid4())
    db = sqlite3.connect(DB_FILE)
    db.execute("INSERT INTO jobs (job_id, status) VALUES (?, 'pending')", (job_id,))
    db.commit()
    text_content, image_prompt = generate_gemini_content(theme)
    image_local_path, image_url = search_taiwan_memory(theme, job_id)
    if not image_url:
        print("流程: 搜尋找不到，退回使用 AI (T2I - API) 生成圖片。")
        image_url, image_local_path = generate_t2i_image(image_prompt, job_id)
    if not all([text_content, image_url, image_local_path]):
        db.execute("UPDATE jobs SET status = 'failed' WHERE job_id = ?", (job_id,))
        db.commit(); db.close()
        return jsonify({"error": "生成初始素材失敗"}), 500
    db.execute(
        "UPDATE jobs SET status = 'generating_video', text_content = ?, image_url = ?, image_prompt_en = ? WHERE job_id = ?",
        (text_content, image_url, image_prompt, job_id)
    )
    db.commit(); db.close()
    thread = threading.Thread(target=run_svd_background_task, args=(job_id, image_local_path))
    thread.start()
    return jsonify({"job_id": job_id, "text_content": text_content, "image_url": image_url})

@app.route('/api/check-status', methods=['GET'])
def check_status():
    # ... (這個檢查狀態的 API 完全不變) ...
    job_id = request.args.get('id')
    # ... (查詢資料庫邏輯不變) ...
    if not job_id: return jsonify({"error": "需要 job_id"}), 400
    db = sqlite3.connect(DB_FILE); cursor = db.cursor()
    cursor.execute("SELECT status, video_url FROM jobs WHERE job_id = ?", (job_id,))
    job = cursor.fetchone(); db.close()
    if not job: return jsonify({"error": "找不到任務"}), 404
    return jsonify({"status": job[0], "video_url": job[1]})

# --- 7. 啟動伺服器 (不變) ---
if __name__ == '__main__':
    # (移除 import numpy as np，除非您其他地方用到)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("已刪除舊的 jobs.db，將建立新版資料庫。")
    init_db()
    app.run(debug=True, port=5000)