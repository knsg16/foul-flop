from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam  # 例としてAdamを使用

# オプティマイザーの設定
optimizer = Adam(learning_rate=0.001)  # ここで必要なパラメータを設定

# モデルのロード
model = load_model("./sample_model.h5", custom_objects={'optimizer': optimizer})
app = Flask(__name__)
app.secret_key = 'secret key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = "static/" + video_file.filename
            
            # static/ディレクトリ内の動画が存在する場合に削除する
            clear_videos()
                
            video_file.save(video_path)
            label, probability = predict_video(video_path)
            session['video_path'] = video_path  # セッションに動画のパスを保存
            flash(f'This video is a {label} with {probability*100:.2f}% confidence!')
            return redirect(url_for('index'))
    video_path = session.get('video_path')  # セッションから動画のパスを取得
    return render_template('index.html', video_path=video_path)

@app.route('/clear_session', methods=['POST'])
def clear_session():
    session.clear()  # セッションの内容をすべて削除
    # static/ディレクトリ内の動画が存在する場合に削除する
    clear_videos()
    return redirect(url_for('index'))  # トップページにリダイレクト

def clear_videos(directory='static'):
    # 指定されたディレクトリ内のすべてのファイルを取得
    files_in_directory = os.listdir(directory)
    
    # 動画ファイルの拡張子をターゲットとしてリストアップ
    filtered_files = [file for file in files_in_directory if file.endswith((".mp4", ".avi", ".mov", ".flv", ".mkv"))]
    
    # ターゲットとなる動画ファイルを削除
    for file in filtered_files:
        os.remove(os.path.join(directory, file))

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (64, 64))
            frames.append(frame_resized)
            if len(frames) == 10:
                input_data = np.expand_dims(frames, axis=0)
                prediction = model.predict(input_data)
                if prediction[0][0] > 0.5:
                    return "foul", prediction[0][0]
                else:
                    return "flop", prediction[0][0]
        else:
            break
    
    cap.release()
    return "undetermined", 0  # 動画が短すぎるか、適切なフレームが取得できなかった場合

if __name__ == "__main__":
    app.secret_key = 'secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run()