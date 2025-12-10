import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import cv2
import face_recognition
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# =================================================================================================
# CẤU HÌNH
# =================================================================================================
app = Flask(__name__)
DATA_DIR = Path("face_data")
USER_IMAGES_DIR = DATA_DIR / "user_images"
USERS_FILE = DATA_DIR / "users.json"
ENCODINGS_FILE = DATA_DIR / "encodings.npy"
LABELS_FILE = DATA_DIR / "labels.json"
RECOGNITION_THRESHOLD = 0.5  # Ngưỡng nhận diện: giá trị càng thấp càng chính xác, nhưng dễ bỏ sót. 0.5 là một khởi đầu tốt.
MAX_COLLECT_IMAGES = 20 # Số lượng ảnh cần thu thập qua webcam

# Tạo các thư mục cần thiết nếu chưa có
DATA_DIR.mkdir(exist_ok=True)
USER_IMAGES_DIR.mkdir(exist_ok=True)

# Khởi tạo face detector của OpenCV để cắt khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =================================================================================================
# CÁC HÀM XỬ LÝ DỮ LIỆU
# =================================================================================================
def read_users():
    """Đọc danh sách người dùng từ file JSON, xử lý an toàn nếu file rỗng hoặc hỏng."""
    if not USERS_FILE.exists():
        return []
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content: return []
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Cảnh báo: File {USERS_FILE} bị hỏng hoặc rỗng. Sẽ tạo lại file mới.")
        return []

def write_users(users):
    """Ghi danh sách người dùng vào file JSON."""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def generate_user_id(name):
    """Tạo ID duy nhất cho người dùng dựa trên tên để tránh trùng lặp."""
    return hashlib.sha256(name.encode('utf-8')).hexdigest()

def update_database():
    """
    Quét toàn bộ ảnh đã lưu, tạo mã hóa (encodings) và lưu lại để nhận diện.
    Đây chính là quá trình "training" lại model.
    """
    print("Bắt đầu quá trình training (cập nhật mã hóa)...")
    users = read_users()
    all_encodings = []
    all_labels = []

    for user in users:
        user_id = user['id']
        user_name = user['name']
        user_dir = USER_IMAGES_DIR / user_id
        if not user_dir.is_dir(): continue

        for img_path in user_dir.glob('*.jpg'):
            try:
                image = face_recognition.load_image_file(str(img_path))
                # Giả định mỗi ảnh chỉ có một khuôn mặt (đã được cắt ở bước trước)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    all_encodings.append(encodings[0])
                    all_labels.append({"id": user_id, "name": user_name})
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

    if not all_encodings:
        print("Không có khuôn mặt nào để training. Tạo file dữ liệu trống.")
        np.save(ENCODINGS_FILE, np.array([]))
        with open(LABELS_FILE, "w") as f: json.dump([], f)
        return

    np.save(ENCODINGS_FILE, np.array(all_encodings))
    with open(LABELS_FILE, "w", encoding='utf-8') as f:
        json.dump(all_labels, f, indent=4, ensure_ascii=False)
    print(f"Training hoàn tất: {len(all_encodings)} mã hóa cho {len(users)} người dùng.")

def process_and_save_faces(uploaded_files, target_dir):
    """
    Phát hiện, cắt và lưu khuôn mặt từ các file được upload.
    Xử lý ảnh trực tiếp trong bộ nhớ để tránh lỗi I/O.
    """
    saved_count = 0
    for file in uploaded_files:
        try:
            in_memory_file = file.read()
            np_array = np.frombuffer(in_memory_file, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None: continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_img = image[y:y+h, x:x+w]
                save_path = target_dir / f"face_{saved_count + 1}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                cv2.imwrite(str(save_path), face_img)
                saved_count += 1
        except Exception as e:
            print(f"Lỗi khi xử lý file upload {secure_filename(file.filename)}: {e}")
    return saved_count

# =================================================================================================
# CÁC ROUTE CỦA FLASK
# =================================================================================================
@app.route('/')
def index():
    """Trang chính, hiển thị giao diện và danh sách người dùng."""
    users = read_users()
    return render_template('index.html', users=users, max_collect=MAX_COLLECT_IMAGES)

@app.route('/register', methods=['POST'])
def register():
    """Xử lý việc đăng ký người dùng mới (cả upload và webcam)."""
    name = request.form.get('name', '').strip()
    if not name:
        return redirect(url_for('index'))

    user_id = generate_user_id(name)
    user_dir = USER_IMAGES_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    # Xử lý từ file upload
    if 'images' in request.files:
        files = request.files.getlist('images')
        if files and files[0].filename:
            saved_count = process_and_save_faces(files, user_dir)
            if saved_count == 0:
                shutil.rmtree(user_dir) # Dọn dẹp nếu không có ảnh hợp lệ
                return redirect(url_for('index'))

    # Thêm thông tin người dùng vào CSDL nếu chưa tồn tại
    users = read_users()
    if user_id not in {u['id'] for u in users}:
        users.append({
            "id": user_id,
            "name": name,
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        write_users(users)

    update_database() # "Train" lại model
    return redirect(url_for('index'))

@app.route('/collect_webcam_image', methods=['POST'])
def collect_webcam_image():
    """Nhận và lưu một ảnh từ webcam."""
    name = request.form.get('name', '').strip()
    file = request.files.get('image')

    if not name or not file:
        return jsonify({'status': 'error', 'message': 'Thiếu tên hoặc ảnh.'})

    in_memory_file = file.read()
    np_array = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'status': 'error', 'message': 'Không đọc được ảnh.'})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) != 1:
        return jsonify({'status': 'no_face', 'message': 'Không phát hiện khuôn mặt hoặc có quá nhiều khuôn mặt.'})
    
    user_id = generate_user_id(name)
    user_dir = USER_IMAGES_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    saved_count = len(list(user_dir.glob('*.jpg')))
    x, y, w, h = faces[0]
    face_img = image[y:y+h, x:x+w]
    save_path = user_dir / f"face_{saved_count + 1}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    cv2.imwrite(str(save_path), face_img)
    
    return jsonify({'status': 'ok', 'saved_count': saved_count + 1})


@app.route('/recognize', methods=['POST'])
def recognize():
    """Nhận diện khuôn mặt từ ảnh webcam."""
    file = request.files.get('image')
    if not file:
        return jsonify({"status": "error", "message": "Không có file ảnh."})

    try:
        known_encodings = np.load(ENCODINGS_FILE)
        if known_encodings.shape[0] == 0:
            return jsonify({"status": "unknown", "message": "CSDL trống."})

        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            known_labels = json.load(f)

        image = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(image)
        unknown_encodings = face_recognition.face_encodings(image, face_locations)

        if not unknown_encodings:
            return jsonify({"status": "no_face"})

        # So sánh khuôn mặt đầu tiên tìm thấy
        unknown_encoding = unknown_encodings[0]
        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]

        if min_distance < RECOGNITION_THRESHOLD:
            matched_user = known_labels[best_match_index]
            return jsonify({
                "status": "success",
                "user": matched_user,
                "distance": f"{min_distance:.4f}"
            })
        else:
            return jsonify({
                "status": "unknown",
                "message": "Người lạ",
                "distance": f"{min_distance:.4f}"
            })

    except Exception as e:
        print(f"Lỗi nhận diện: {e}")
        return jsonify({"status": "error", "message": "Lỗi hệ thống."})

@app.route('/delete_user/<user_id>', methods=['POST'])
def delete_user(user_id):
    """Xóa người dùng và tất cả dữ liệu liên quan."""
    users = read_users()
    users = [u for u in users if u['id'] != user_id]
    write_users(users)
    
    user_dir = USER_IMAGES_DIR / user_id
    if user_dir.is_dir():
        shutil.rmtree(user_dir)
        
    update_database() # "Train" lại model sau khi xóa
    return redirect(url_for('index'))

# =================================================================================================
# CHẠY ỨNG DỤNG
# =================================================================================================
if __name__ == '__main__':
    # "Train" lại model một lần khi khởi động để đảm bảo dữ liệu luôn mới nhất
    if not ENCODINGS_FILE.exists() or not LABELS_FILE.exists() or not USERS_FILE.exists():
        update_database()
    app.run(debug=True, host='0.0.0.0', port=5001) 