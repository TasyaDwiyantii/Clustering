import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, request, render_template, url_for, redirect, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Gunakan kunci rahasia untuk sesi

UPLOAD_FOLDER = 'static/uploads'
SEGMENTED_FOLDER = 'static/segmented'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

# Fungsi untuk segmentasi gambar
def segment_image(image_path, n_clusters=3):
    image = Image.open(image_path)
    image_array = np.array(image)  # Gambar asli

    # Flatten gambar
    image_array_reshaped = image_array.reshape((-1, 3))

    # Clustering menggunakan KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(image_array_reshaped)

    # Rekonstruksi gambar
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image_array.shape).astype(np.uint8)

    segmented_image = Image.fromarray(segmented_image)
    return segmented_image

@app.route('/')
def index():
    # Ambil data gambar yang disimpan dalam sesi (jika ada)
    original_image = session.get('original_image', None)
    segmented_image = session.get('segmented_image', None)
    n_clusters = session.get('n_clusters', 3)

    # Jika gambar sudah diproses, menghapusnya setelah refresh
    if original_image and segmented_image:
        session.pop('original_image', None)
        session.pop('segmented_image', None)
        session.pop('n_clusters', None)

    return render_template('index.html', original_image=original_image, 
                           segmented_image=segmented_image, n_clusters=n_clusters)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    n_clusters = request.form.get('n_clusters', 3)

    if file and file.filename:
        input_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_image_path)

        try:
            n_clusters = int(n_clusters)
        except ValueError:
            return 'Please enter a valid number for clusters.', 400

        segmented_image = segment_image(input_image_path, n_clusters)

        output_image_path = os.path.join(SEGMENTED_FOLDER, f"segmented_{file.filename}")
        segmented_image.save(output_image_path)

        # Menyimpan URL gambar di session
        original_image_url = url_for('static', filename=f'uploads/{file.filename}')
        segmented_image_url = url_for('static', filename=f'segmented/segmented_{file.filename}')

        session['original_image'] = original_image_url
        session['segmented_image'] = segmented_image_url
        session['n_clusters'] = n_clusters

        return redirect('/')  # Kembali ke halaman utama setelah pemrosesan

if __name__ == '__main__':
    app.run(debug=True)
