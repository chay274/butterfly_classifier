from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Load your trained model
model = load_model("butterfly_model_v1.h5")

# Load class labels from the training data
train_dir = "./dataset/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class_labels = list(train_generator.class_indices.keys())

# Home route: show index page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded image
        file = request.files["file"]
        if file:
            # Ensure uploads folder exists
            uploads_folder = os.path.join("static", "uploads")
            os.makedirs(uploads_folder, exist_ok=True)

            # Save file into static/uploads
            file_path = os.path.join(uploads_folder, file.filename)
            file.save(file_path)

            # Preprocess image for prediction
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)

            # Make prediction
            prediction = model.predict(x)
            predicted_class_index = np.argmax(prediction)
            predicted_species = class_labels[predicted_class_index]
            confidence = f"{np.max(prediction) * 100:.2f}%"

            # Return result page with prediction
            return render_template("result.html",
                                   species=predicted_species,
                                   confidence=confidence,
                                   uploaded_filename=file.filename)

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

