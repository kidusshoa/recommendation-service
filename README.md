# recommendation-service

recommendation engine for khanut-local-business-finder

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install fastapi uvicorn pandas scikit-learn surprise joblib
```

or

```bash
pip install -r requirements.txt
```

#to start

```bash
uvicorn app.main:app --reload --port 5000
```

# API - endpoints

```bash
GET http://localhost:5000/recommendations/u001
```

```bash
POST http://localhost:5000/retrain
```
