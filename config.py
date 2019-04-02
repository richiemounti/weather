import os


class Config:
    DEBUG = False
    FLASK_DEBUG = 0
    SECRET_KEY = "I am the King"



class DevelopmentConfig(Config):
    FLASK_DEBUG = 1
    DEBUG = True



class TestConfig(Config):
    FLASK_DEBUG = 1
    DEBUG = True
    TESTING = True
  
configs = dict(
    testing = TestConfig,
    production=Config,
    development=DevelopmentConfig
)