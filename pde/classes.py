"""Class list, PlantVillage <-> PlantDoc name mapping, and the advice line shown in the app.

The order of CLASSES is the model's output order. Do not reorder after training.
"""

# Canonical class names = PlantVillage tomato folder names.
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Short names for plots, MQTT payloads and the app.
SHORT = {
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___healthy": "healthy",
}
SHORT_CLASSES = [SHORT[c] for c in CLASSES]

# PlantDoc folder name -> canonical class. PlantDoc has no Target Spot class,
# so the field test set has 9 of the 10 classes.
PLANTDOC_TO_CLASS = {
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf": "Tomato___healthy",
}

# One short advice line per class (general guidance, not a substitute for an agronomist).
ADVICE = {
    "bacterial_spot": "Remove spotted leaves, avoid overhead watering; copper spray can slow spread.",
    "early_blight": "Remove lower infected leaves, mulch the soil, rotate crops; consider a fungicide.",
    "late_blight": "Act fast: remove and bag infected plants, keep foliage dry; spreads quickly in cool wet weather.",
    "leaf_mold": "Improve airflow and lower humidity (ventilate, space plants); remove affected leaves.",
    "septoria_leaf_spot": "Remove infected lower leaves, avoid wetting foliage, rotate crops next season.",
    "spider_mites": "Spray leaf undersides with water or insecticidal soap; check nearby plants.",
    "target_spot": "Remove infected leaves, improve airflow, avoid overhead irrigation.",
    "yellow_leaf_curl_virus": "Virus spread by whiteflies: remove infected plants, control whiteflies, use resistant varieties.",
    "mosaic_virus": "No cure: remove infected plants, disinfect tools and hands, avoid tobacco near plants.",
    "healthy": "Leaf looks healthy. Keep monitoring regularly.",
}

NOT_SURE_MSG = "Not sure - please retake the photo (one leaf, fill the frame, good light)."
