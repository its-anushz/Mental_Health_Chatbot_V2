import os

class Config:
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "my_secret_key_here")  # default fallback key
    SQLALCHEMY_DATABASE_URI = 'sqlite:///mental_health_chatbot.db'   # local SQLite file
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")   # fallback value
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")    # fallback value
