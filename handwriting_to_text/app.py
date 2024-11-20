from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import numpy as np
import tensorflow as tf
from keras import ops
import keras
from keras.layers import StringLookup
import os
import cv2
from tensorflow.keras.preprocessing.image import save_img
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'E:\5th Sem\SDP\personalized\handwriting_to_text\uploads'
app.config['PROCESSED_FOLDER'] = r'E:\5th Sem\SDP\personalized\handwriting_to_text\processed'

# Ensure the upload and processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load the pre-trained model
prediction_model = tf.keras.models.load_model('font.h5')

# Model preprocessing and prediction utilities
image_width = 128
image_height = 32

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = ops.cast(image, tf.float32) / 255.0
    return image

base_path = "data"
words_list = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
        words_list.append(line)

len(words_list)

split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)

np.random.shuffle(words_list)

base_image_path = os.path.join(base_path, "words")


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))


AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][:, :50]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
        )
        output_text.append(res)
    return output_text

def sort_bounding_boxes(word_boxes, line_threshold=15):
    """
    Sort bounding boxes in a top-to-bottom, left-to-right reading order.
    :param word_boxes: List of bounding boxes [(x, y, w, h), ...].
    :param line_threshold: Maximum vertical distance between words in the same line.
    :return: Sorted list of bounding boxes.
    """
    # Sort by y-coordinate first to group lines
    word_boxes = sorted(word_boxes, key=lambda box: box[1])

    # Group words into lines based on vertical proximity
    lines = []
    current_line = [word_boxes[0]]

    for i in range(1, len(word_boxes)):
        _, y, _, h = word_boxes[i]
        _, prev_y, _, prev_h = word_boxes[i - 1]

        # If the y-coordinate is close enough to the previous word, consider it the same line
        if abs(y - prev_y) < line_threshold:
            current_line.append(word_boxes[i])
        else:
            # Sort the current line by x-coordinate
            lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [word_boxes[i]]

    # Add the last line
    lines.append(sorted(current_line, key=lambda box: box[0]))

    # Flatten the list of lines back into a single sorted list of bounding boxes
    sorted_word_boxes = [box for line in lines for box in line]
    return sorted_word_boxes


def segment_words_in_paragraph(image_path, dilation_kernel=(10, 3), line_threshold=20, merge_threshold=10):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to get a binary image (inverted)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply dilation to connect letters within words but avoid merging separate words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)

    # Find contours of the words
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    word_images = []
    word_boxes = []

    # Iterate over contours and crop out the words
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)

        # Filter out noise and small boxes
        aspect_ratio = w / h
        if h < 10 or w < 10 or aspect_ratio > 10:
            continue

        word_boxes.append((x, y, w, h))

    # Sort the bounding boxes to ensure logical reading order
    word_boxes = sort_bounding_boxes(word_boxes, line_threshold=15)
    word_boxes = merge_close_contours(word_boxes, threshold=merge_threshold)


    # Extract word images based on bounding boxes
    for (x, y, w, h) in word_boxes:
        word_img = img[y:y + h, x:x + w]
        word_images.append(word_img)

    return word_images, word_boxes

def merge_close_contours(word_boxes, threshold=10):
    merged_boxes = []
    for box in word_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            x, y, w, h = box
            
            # Check if the current box is close to the previous one horizontally
            # Adding a more relaxed condition to avoid merging separate words
            if (x - (prev_x + prev_w)) < threshold and (abs(y - prev_y) < 5):
                # Merge the boxes horizontally
                new_x = min(prev_x, x)
                new_y = min(prev_y, y)
                new_w = max(prev_x + prev_w, x + w) - new_x
                new_h = max(prev_y + prev_h, y + h) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)
    return merged_boxes


def create_temp_folder():
    """Create a temporary folder to store word images."""
    temp_folder = "temp_word_images"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    return temp_folder

def delete_temp_folder(folder_path):
    """Delete the temporary folder after processing."""
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(folder_path)

def save_segmented_images(word_images, temp_folder):
    """Save each segmented word image to the temporary folder."""
    word_paths = []
    for i, word_img in enumerate(word_images):
        # Ensure the word image has 3 dimensions (height, width, channels)
        word_img = np.expand_dims(word_img, axis=-1)  # Add a channel dimension
        
        # Save the word image
        word_path = os.path.join(temp_folder, f"word_{i}.png")
        save_img(word_path, word_img)  # Save the image using Keras save_img
        word_paths.append(word_path)
    return word_paths

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('process_image', filename=file.filename))
    return render_template('upload.html')

@app.route('/process/<filename>')
def process_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    word_images, word_boxes = segment_words_in_paragraph(file_path)
    temp_folder = create_temp_folder()
    word_paths = save_segmented_images(word_images, temp_folder)
    predicted_words = []
    for i, word_path in enumerate(word_paths):
        processed_word_img = preprocess_image(word_path)  # Process the image from the temp folder
        processed_word_img = np.expand_dims(processed_word_img, axis=0)  # Add batch dimension
        prediction = prediction_model.predict(processed_word_img)
        decoded_text = decode_batch_predictions(prediction)
        predicted_words.append(decoded_text[0])
    predicted_text = ' '.join(predicted_words)
    predicted_text = predicted_text.replace("[UNK]", '')
    delete_temp_folder(temp_folder)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(processed_path, cv2.imread(file_path))
    return render_template('result.html', image_url=filename, text=predicted_text)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
