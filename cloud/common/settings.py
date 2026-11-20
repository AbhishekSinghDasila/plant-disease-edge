import os

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://plant:plant@postgres:5432/plant")
SERVICE = os.getenv("SERVICE_NAME", "service")
TOPIC_PREFIX = os.getenv("TOPIC_PREFIX", "plant")
