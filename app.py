import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imagehash
import collections
import base64
from io import BytesIO

# 페이지 설정
st.set_page_config(layout="wide", page_title="Photo Cleaner Pro", initial_sidebar_state="collapsed")

# ==========================================
# 🎛️ [튜닝 컨트롤 패널] 설정값을 여기서 변경하세요
# ==========================================

# 1. 유사도 민감도 (정수)
# - 값이 클수록 '덜 비슷한' 사진도 같은 그룹으로 묶습니다.
# - 3~4: 매우 엄격함 (거의 똑같아야 함)
# - 5~6: 적당함 (미세한 움직임 허용) - 추천!
# - 7~: 느슨함 (구도가 달라도 묶일 수 있음)
SIMILARITY_THRESHOLD = 5 

# 2. 블러(흐림) 경고 기준 (실수)
# - 이 점수보다 낮으면 '심각하게 흐림'으로 판단할 수 있습니다.
# - (참고: 이 앱은 절대 수치보다 그룹 내 상대 평가를 우선합니다)
BLUR_THRESHOLD = 100.0

# ==========================================

# --- 1. 콜백 함수 추가  ---
def toggle_state(key):
    """버튼 클릭 시 체크박스 상태를 반전시키는 콜백 함수"""
    st.session_state[key] = not st.session_state[key]

# --- 2. 기존 스타일 및 유틸리티 함수 (그대로 유지) ---
st.markdown("""
<style>
    .photo-card {
        border-radius: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        width: 100%;
        display: block;
    }
    .photo-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        z-index: 10;
    }
    .border-keep { border: 4px solid #4CAF50; }
    .border-delete { border: 4px solid #FF4B4B; opacity: 0.6; }
    .caption-text {
        font-size: 0.9rem;
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def img_to_base64(img_pil):
    img_pil = img_pil.copy()
    img_pil.thumbnail((300, 300))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def get_image_quality(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_images(uploaded_files):
    image_data = []
    for file in uploaded_files:
        file_size = file.size
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_pil = Image.open(file)
        
        img_hash = imagehash.phash(img_pil)
        score = get_image_quality(img_cv)
        
        image_data.append({
            "file": file,
            "hash": img_hash,
            "score": score,
            "size": file_size,
            "name": file.name,
            "preview": img_pil,
            "base64": img_to_base64(img_pil)
        })
        file.seek(0) 

    groups = collections.defaultdict(list)
    processed_indices = set()

    for i in range(len(image_data)):
        if i in processed_indices: continue
        current_img = image_data[i]
        group_id = str(current_img['hash']) 
        groups[group_id].append(current_img)
        processed_indices.add(i)

        for j in range(i + 1, len(image_data)):
            if j in processed_indices: continue
            compare_img = image_data[j]
            if current_img['hash'] - compare_img['hash'] <= 5:
                groups[group_id].append(compare_img)
                processed_indices.add(j)

    sorted_groups = []
    for group_id, items in groups.items():
        items.sort(key=lambda x: (x['size'], x['score']), reverse=True)
        sorted_groups.append(items)
    
    return sorted_groups

# --- UI 구성 ---

st.title("📸 Photo Cleaner Pro")
# --- 사용자 안내 메시지 ---
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
    <h5 style="margin-top: 0;">💡 사용 방법</h5>
    <ul style="line-height: 1.6;">
        <li>📸 <b>자동 분석:</b> 중복되거나 초점이 흐린 사진을 자동으로 찾아냅니다.</li>
        <li>
            <span style="color: #FF4B4B; font-weight: bold; border: 2px solid #FF4B4B; padding: 2px 6px; border-radius: 4px;">🟥 붉은 테두리</span>
            : <b>삭제될 사진</b>입니다. (일괄 삭제 시 지워집니다)
        </li>
        <li>
            <span style="color: #4CAF50; font-weight: bold; border: 2px solid #4CAF50; padding: 2px 6px; border-radius: 4px;">🟩 초록 테두리</span>
            : <b>남길 사진</b>입니다.
        </li>
        <li>🖱️ <b>선택 변경:</b> 사진 아래 <b>[살리기/지우기]</b> 버튼을 누르면 상태가 바뀝니다.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
# -------------------------------------

uploaded_files = st.file_uploader("갤러리 사진 업로드 (다중 선택)", 
                                  type=['jpg', 'jpeg', 'png'], 
                                  accept_multiple_files=True)

if uploaded_files:
    if "grouped_photos" not in st.session_state or st.button("🔄 사진 다시 분석하기"):
        with st.spinner('AI가 사진을 분석하고 분류 중입니다...'):
            grouped_photos = process_images(uploaded_files)
            st.session_state['grouped_photos'] = grouped_photos
            for group in grouped_photos:
                for i, photo in enumerate(group):
                    key = f"del_{photo['name']}"
                    if key not in st.session_state:
                        # 첫 번째 사진은 유지(False), 나머지는 삭제(True)
                        st.session_state[key] = (i != 0)

# 결과 화면
if 'grouped_photos' in st.session_state:
    groups = st.session_state['grouped_photos']
    
    total_deleted_size = 0
    total_deleted_count = 0
    final_delete_list = []

    st.divider()
    
    for idx, group in enumerate(groups):
        if len(group) == 1: continue 
            
        st.subheader(f"📂 그룹 #{idx+1}")
        
        cols = st.columns(min(len(group), 4))
        
        for i, photo in enumerate(group):
            col_idx = i % 4
            size_mb = photo['size'] / (1024 * 1024)
            key = f"del_{photo['name']}"
            
            # 현재 상태 확인 (체크박스는 UI상에 안 보이지만 상태값은 가짐)
            is_deleted = st.session_state[key]
            
            with cols[col_idx]:
                # 1. 상태값 저장을 위한 숨겨진 체크박스 (label_visibility="collapsed")
                st.checkbox("삭제", key=key, label_visibility="collapsed")
                
                # 2. 시각적 표현 (HTML/CSS)
                border_class = "border-delete" if is_deleted else "border-keep"
                status_text = "🗑️ DELETE" if is_deleted else "✅ KEEP"
                status_color = '#FF4B4B' if is_deleted else '#4CAF50'
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="{photo['base64']}" class="photo-card {border_class}">
                    <div class="caption-text" style="color: {status_color};">
                        {status_text}<br>
                        <span style="color: gray; font-size: 0.8rem;">{size_mb:.2f} MB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. 토글 버튼 (on_click 사용으로 에러 해결!)
                btn_label = "살리기 💚" if is_deleted else "지우기 🗑️"
                btn_type = "secondary" if is_deleted else "primary"
                
                # 핵심 변경: args=(key,)를 통해 어떤 키를 바꿀지 콜백에 전달
                st.button(btn_label, key=f"btn_{photo['name']}", 
                          on_click=toggle_state, args=(key,), 
                          type=btn_type, use_container_width=True)

                # 통계 집계
                if is_deleted:
                    total_deleted_size += photo['size']
                    total_deleted_count += 1
                    final_delete_list.append(photo['name'])
        
        st.divider()

    # --- 하단 삭제 리포트 ---
    saved_mb = total_deleted_size / (1024 * 1024)
    
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.info(f"선택된 **{total_deleted_count}장** 삭제 시, 약 **{saved_mb:.2f} MB** 확보 가능")
    
    with col_r:
        if st.button("🚨 일괄 삭제 실행", type="primary", use_container_width=True):
            if total_deleted_count > 0:
                st.balloons()
                st.success(f"{total_deleted_count}장 정리 완료!")
                st.markdown(f"""
                <div style="padding: 15px; background-color: rgba(0, 128, 0, 0.1); border-radius: 10px;">
                    <ul>
                        <li>삭제 수량: <b>{total_deleted_count}장</b></li>
                        <li>확보 용량: <b>{saved_mb:.2f} MB</b></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("삭제할 사진이 없습니다.")