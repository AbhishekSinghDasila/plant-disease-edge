"""All 38 PlantVillage classes (14 crops) in the output order of the 38-class models.

Order = byte-sorted PlantVillage folder names (the order Keras' image_dataset_from_directory uses),
checked 2026-11-17 against the pretrained model on 30 images per class (data/pv_allcrops).

Each class id maps to crop, disease, a short machine id, and a one-line offline fallback advice.
Full measures (symptoms, actions, prevention) live in the cloud advisor knowledge base
(cloud/advisor/kb.yaml). This file is the offline minimum the Pi needs.
"""

PV38 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
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

# folder name -> (crop id, disease id, display crop, display disease, offline advice)
_INFO = {
    "Apple___Apple_scab": ("apple", "apple_scab", "Apple", "Apple scab",
        "Fungal: rake and destroy fallen leaves, prune for airflow; protectant fungicide from bud break in wet springs."),
    "Apple___Black_rot": ("apple", "black_rot", "Apple", "Black rot",
        "Fungal: cut out cankers and mummified fruit, remove dead wood; fungicide cover sprays in wet weather."),
    "Apple___Cedar_apple_rust": ("apple", "cedar_apple_rust", "Apple", "Cedar apple rust",
        "Fungal: remove nearby juniper/cedar galls if possible; fungicide at pink bud to petal fall; resistant varieties."),
    "Apple___healthy": ("apple", "healthy", "Apple", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Blueberry___healthy": ("blueberry", "healthy", "Blueberry", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Cherry_(including_sour)___Powdery_mildew": ("cherry", "powdery_mildew", "Cherry", "Powdery mildew",
        "Fungal: prune for airflow, avoid excess nitrogen; sulfur or potassium bicarbonate sprays at first signs."),
    "Cherry_(including_sour)___healthy": ("cherry", "healthy", "Cherry", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn", "gray_leaf_spot", "Corn", "Gray leaf spot (Cercospora)",
        "Fungal: rotate crops, bury or remove residue, resistant hybrids; fungicide if it reaches upper leaves before tasseling."),
    "Corn_(maize)___Common_rust_": ("corn", "common_rust", "Corn", "Common rust",
        "Fungal: usually minor; plant resistant hybrids; fungicide only if severe early in the season."),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "northern_leaf_blight", "Corn", "Northern leaf blight",
        "Fungal: resistant hybrids, rotate, manage residue; fungicide at early tasseling if lesions spread upward."),
    "Corn_(maize)___healthy": ("corn", "healthy", "Corn", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Grape___Black_rot": ("grape", "black_rot", "Grape", "Black rot",
        "Fungal: remove mummified berries and infected canes, open the canopy; fungicide from early shoot growth."),
    "Grape___Esca_(Black_Measles)": ("grape", "esca", "Grape", "Esca (black measles)",
        "Trunk disease: no cure; prune out dead wood in dry weather, protect pruning wounds, remove badly affected vines."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("grape", "leaf_blight", "Grape", "Leaf blight (Isariopsis)",
        "Fungal: remove infected leaves, improve airflow; the same fungicide program as for other grape leaf diseases."),
    "Grape___healthy": ("grape", "healthy", "Grape", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Orange___Haunglongbing_(Citrus_greening)": ("orange", "citrus_greening", "Orange", "Huanglongbing (citrus greening)",
        "Bacterial, spread by psyllids, no cure: report to local extension, control psyllids, remove infected trees."),
    "Peach___Bacterial_spot": ("peach", "bacterial_spot", "Peach", "Bacterial spot",
        "Bacterial: resistant varieties, avoid overhead watering; copper sprays in dormancy/early season."),
    "Peach___healthy": ("peach", "healthy", "Peach", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Pepper,_bell___Bacterial_spot": ("pepper", "bacterial_spot", "Bell pepper", "Bacterial spot",
        "Bacterial: remove infected leaves, avoid wetting foliage, rotate 2-3 years; copper sprays slow the spread."),
    "Pepper,_bell___healthy": ("pepper", "healthy", "Bell pepper", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Potato___Early_blight": ("potato", "early_blight", "Potato", "Early blight",
        "Fungal: remove lower infected leaves, keep plants well fed and watered at the base; rotate; fungicide if spreading."),
    "Potato___Late_blight": ("potato", "late_blight", "Potato", "Late blight",
        "Act fast: remove and bag infected plants, keep foliage dry; spreads very quickly in cool wet weather."),
    "Potato___healthy": ("potato", "healthy", "Potato", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Raspberry___healthy": ("raspberry", "healthy", "Raspberry", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Soybean___healthy": ("soybean", "healthy", "Soybean", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Squash___Powdery_mildew": ("squash", "powdery_mildew", "Squash", "Powdery mildew",
        "Fungal: remove worst leaves, space plants for airflow; sulfur, potassium bicarbonate or neem at first signs."),
    "Strawberry___Leaf_scorch": ("strawberry", "leaf_scorch", "Strawberry", "Leaf scorch",
        "Fungal: remove infected leaves after harvest, avoid overhead watering, renew beds every few years."),
    "Strawberry___healthy": ("strawberry", "healthy", "Strawberry", "Healthy", "Leaf looks healthy. Keep monitoring."),
    "Tomato___Bacterial_spot": ("tomato", "bacterial_spot", "Tomato", "Bacterial spot",
        "Remove spotted leaves, avoid overhead watering; copper spray can slow spread."),
    "Tomato___Early_blight": ("tomato", "early_blight", "Tomato", "Early blight",
        "Remove lower infected leaves, mulch the soil, rotate crops; consider a fungicide."),
    "Tomato___Late_blight": ("tomato", "late_blight", "Tomato", "Late blight",
        "Act fast: remove and bag infected plants, keep foliage dry; spreads quickly in cool wet weather."),
    "Tomato___Leaf_Mold": ("tomato", "leaf_mold", "Tomato", "Leaf mold",
        "Improve airflow and lower humidity (ventilate, space plants); remove affected leaves."),
    "Tomato___Septoria_leaf_spot": ("tomato", "septoria_leaf_spot", "Tomato", "Septoria leaf spot",
        "Remove infected lower leaves, avoid wetting foliage, rotate crops next season."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "spider_mites", "Tomato", "Spider mites",
        "Spray leaf undersides with water or insecticidal soap; check nearby plants."),
    "Tomato___Target_Spot": ("tomato", "target_spot", "Tomato", "Target spot",
        "Remove infected leaves, improve airflow, avoid overhead irrigation."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "yellow_leaf_curl_virus", "Tomato", "Yellow leaf curl virus",
        "Virus spread by whiteflies: remove infected plants, control whiteflies, use resistant varieties."),
    "Tomato___Tomato_mosaic_virus": ("tomato", "mosaic_virus", "Tomato", "Mosaic virus",
        "No cure: remove infected plants, disinfect tools and hands, avoid tobacco near plants."),
    "Tomato___healthy": ("tomato", "healthy", "Tomato", "Healthy", "Leaf looks healthy. Keep monitoring."),
}
assert set(_INFO) == set(PV38) and len(PV38) == 38


def class_id(folder):
    """'Tomato___Late_blight' -> 'tomato.late_blight' (stable id used in MQTT, DB and the advisor KB)."""
    crop, disease = _INFO[folder][:2]
    return f"{crop}.{disease}"


def info(folder):
    crop, disease, crop_name, disease_name, advice = _INFO[folder]
    return {"id": f"{crop}.{disease}", "folder": folder, "crop": crop, "disease": disease,
            "crop_name": crop_name, "disease_name": disease_name, "healthy": disease == "healthy",
            "display": f"{crop_name}: {disease_name}", "advice": advice}


PV38_IDS = [class_id(f) for f in PV38]
BY_ID = {class_id(f): info(f) for f in PV38}
CROPS = sorted({i["crop"] for i in BY_ID.values()})
