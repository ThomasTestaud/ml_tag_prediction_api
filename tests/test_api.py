import pytest
from flask import Flask
from api import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_api_route(client):
    response = client.get('/api')
    assert response.status_code == 200
    assert response.json == {"message": "Hello World!"}

def test_predict_tags_route(client):
    response = client.post('/api/predict-tags', json={
        "title": "How to use Flask?",
        "body": "I need help with Flask routing."
    })
    assert response.status_code == 200
    assert "tags" in response.json
