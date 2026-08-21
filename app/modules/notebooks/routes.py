from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

notebooks_bp = Blueprint('notebooks', __name__)

UPLOAD_FOLDER = '/path/to/notebook/scoped/storage'
ALLOWED_EXTENSIONS = {'csv', 'tsv', 'json', 'txt', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@notebooks_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_id = request.headers.get('X-User-ID')  # Example of user scoping
        notebook_id = request.headers.get('X-Notebook-ID')

        user_folder = os.path.join(UPLOAD_FOLDER, user_id, notebook_id)
        os.makedirs(user_folder, exist_ok=True)

        file_path = os.path.join(user_folder, filename)
        file.save(file_path)

        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200

    return jsonify({'error': 'Invalid file type'}), 400