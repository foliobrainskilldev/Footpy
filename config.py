from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Football ML API"
    environment: str = "production"
    cors_origin: str = "*"
    model_save_path: str = "./models"
    default_edge_threshold: float = 0.05
    initial_bankroll: float = 1000.0

    class Config:
        env_file = ".env"

settings = Settings()