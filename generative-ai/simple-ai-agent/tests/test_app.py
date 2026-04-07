import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestApp:
    def test_app_title(self):
        assert app.title == "AI Agent API"
        assert app.version == "1.0.0"
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "AI Agent API is running"}
    
    def test_health_route_included(self, client):
        response = client.get("/health/")
        assert response.status_code == 200
    
    def test_chat_route_included(self, client):
        response = client.post("/chat/", json={"query": "test"})
        assert response.status_code in [200, 500]
    
    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "AI Agent API"
        assert schema["info"]["version"] == "1.0.0"
    
    def test_docs_endpoint(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
