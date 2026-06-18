import os

_TEST_ENV = {
    "ENV": "test",
    "REDIS_URL": "redis://localhost:6379/0",
    "RABBITMQ_URL": "amqp://localhost:5672/",
    "JWT_SECRET": "test-jwt-secret",
    "INTERNAL_SECRET": "test-internal-secret",
    "GOOGLE_PROJECT_ID": "test-project",
    "GEMINI_API_KEY": "test-gemini-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "ELEVENLABS_API_KEY": "test-elevenlabs-key",
    "ELEVENLABS_VOICE_ID": "test-voice-id",
    "SPRING_BOOT_INTERNAL_URL": "http://localhost:8080",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)
