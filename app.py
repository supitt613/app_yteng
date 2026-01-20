import streamlit as st
import yt_dlp
import whisper
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import tempfile
import shutil

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Helper Functions ---

@st.cache_resource
def load_whisper_model(model_name="base"):
    """Loads the Whisper model, caching it for reuse."""
    return whisper.load_model(model_name)

def download_youtube_audio(url):
    """Downloads audio from a YouTube URL to a temporary file."""
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': audio_path,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', 'Unknown Title')
            return audio_path, video_title, temp_dir
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, None, None

def transcribe_audio(audio_path, model):
    """Transcribes audio using the Whisper model."""
    result = model.transcribe(audio_path)
    return result['text']

def clean_text_for_chunks(text):
    """Removes non-alphanumeric characters except spaces and converts to lowercase."""
    # Keep only letters, numbers, and spaces. Remove extra spaces.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_language_chunks(text, num_chunks=7, min_len=2, max_len=4):
    """
    Extracts common N-gram language chunks from text.
    Filters for meaningful chunks (not just stop words, not starting/ending with stop words).
    """
    stop_words = set(stopwords.words('english'))
    cleaned_text = clean_text_for_chunks(text)
    words = word_tokenize(cleaned_text)

    all_candidate_chunks = []
    for n in range(min_len, max_len + 1):
        for i in range(len(words) - n + 1):
            chunk_words = words[i:i+n]
            chunk_str = " ".join(chunk_words)

            # Filter criteria for meaningful chunks:
            # 1. Contains at least one non-stop word
            # 2. Does not start or end with a stop word
            if (not any(word not in stop_words for word in chunk_words) or # If all are stop words
                chunk_words[0] in stop_words or
                chunk_words[-1] in stop_words):
                continue

            all_candidate_chunks.append(chunk_str)

    # Count frequency of candidate chunks
    chunk_counts = Counter(all_candidate_chunks)

    # Sort by frequency (descending)
    sorted_chunks = sorted(chunk_counts.items(), key=lambda item: item[1], reverse=True)

    # Select the top N chunks
    final_chunks = [chunk for chunk, _ in sorted_chunks[:num_chunks]]

    return final_chunks

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="YT Easy English Chunk Extractor")

st.title("🗣️ YT Easy English 語塊擷取器")
st.markdown("""
    輸入 YouTube 影片連結，此應用程式將會：
    1. 從影片中擷取音訊。
    2. 將音訊轉錄成文字。
    3. 將轉錄文本分割成對話片段。
    4. 從每個對話片段中找出 6-8 個常用的英語語塊 (phrases) 供您練習。
    
    **注意：** 此應用程式需要您的系統安裝 `ffmpeg`。您可以從 [ffmpeg.org](https://ffmpeg.org/download.html) 下載並安裝。
    """)

youtube_url = st.text_input("請輸入 YouTube 影片連結 (例如: `https://www.youtube.com/watch?v=k_B_t1_d_24`) ", "")

if youtube_url:
    if "youtube.com/watch?v=" not in youtube_url and "youtu.be/" not in youtube_url:
        st.error("請輸入有效的 YouTube 影片連結。")
    else:
        st.video(youtube_url) # Display the video directly

        st.subheader("處理中...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. Download Audio
        status_text.text("1/3 正在下載音訊...")
        progress_bar.progress(33)
        audio_path, video_title, temp_dir = download_youtube_audio(youtube_url)

        if audio_path:
            st.success(f"已成功下載影片: **{video_title}**")
            st.markdown(f"---")
            st.subheader(f"影片標題: {video_title}")

            # 2. Transcribe Audio
            status_text.text("2/3 正在轉錄音訊 (這可能需要一些時間)... ")
            progress_bar.progress(66)
            try:
                model = load_whisper_model("base") # Using 'base' model for faster processing
                full_transcript = transcribe_audio(audio_path, model)
                st.success("音訊轉錄完成！")
            except Exception as e:
                st.error(f"轉錄音訊時發生錯誤: {e}")
                full_transcript = None
            finally:
                # Clean up temporary audio file and directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            if full_transcript:
                status_text.text("3/3 正在分析語塊...")
                progress_bar.progress(100)

                st.subheader("完整轉錄文本 (可選)")
                with st.expander("點擊查看完整轉錄文本"):
                    st.write(full_transcript)

                st.subheader("對話片段與常用語塊")

                # Segment the full transcript into sentences
                sentences = sent_tokenize(full_transcript)

                # Define dialogue segments (e.g., 3-5 sentences per segment)
                segment_size = 4 # Average number of sentences per segment
                num_chunks_per_segment = 7 # User requested 6-8

                dialogue_segments = []
                for i in range(0, len(sentences), segment_size):
                    segment_text = " ".join(sentences[i:i+segment_size])
                    dialogue_segments.append(segment_text)

                if not dialogue_segments:
                    st.warning("未能從轉錄文本中分割出對話片段。")
                else:
                    for i, segment in enumerate(dialogue_segments):
                        st.markdown(f"#### 對話片段 {i+1}")
                        st.info(segment) # Display the dialogue segment

                        # Extract chunks for this segment
                        chunks = extract_language_chunks(segment, num_chunks=num_chunks_per_segment)

                        if chunks:
                            st.markdown("**建議練習語塊:**")
                            cols = st.columns(3)
                            for j, chunk in enumerate(chunks):
                                cols[j % 3].success(f"👉 {chunk}")
                        else:
                            st.warning("未能從此片段中找到常用語塊。")
                        st.markdown("---")
            else:
                st.error("無法進行語塊分析，因為轉錄失敗。")
        else:
            st.error("無法進行語塊分析，因為音訊下載失敗。")

        progress_bar.empty()
        status_text.empty()
