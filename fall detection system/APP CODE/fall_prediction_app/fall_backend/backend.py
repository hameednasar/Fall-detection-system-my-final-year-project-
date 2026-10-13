
from __future__ import annotations
import os, bcrypt, numpy as np, joblib
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

SECRET_KEY  = os.getenv("SECRET_KEY", "fallpredict_secret_key_change_in_prod")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   

MODEL_PATH  = "fall_detection_model.keras"
SCALER_PATH = "fall_detection_scaler.pkl"

WINDOW_SIZE = 150   
N_FEATURES  = 6     

SENSOR_TIMEOUT_SEC = 8   
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain[:72].encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain[:72].encode(), hashed.encode())

_model  = None
_scaler = None

def _load_model():
    global _model, _scaler
    if _model is None:
        try:
            import tensorflow as tf
            _model  = tf.keras.models.load_model(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            print(f"  Model loaded  –  input shape: {_model.input_shape}")
        except Exception as e:
            print(f"   Model load failed: {e}")
    return _model, _scaler

_DEMO_EMAIL    = "hamza@gmail.com"
_DEMO_PASSWORD = "kk123456"
_DEMO_NAME     = "Hamza"

_users: dict[str, dict] = {
    _DEMO_EMAIL: {
        "name": _DEMO_NAME,
        "hashed_password": hash_password(_DEMO_PASSWORD),
    }
}

_latest_status: dict = {
    "fall_detected":    False,
    "confidence":       0.0,
    "risk_label":       "LOW",
    "risk_score":       0,
    "timestamp":        datetime.now(timezone.utc).isoformat(),
    "sensor_connected": False,
    "rows_collected":   0,
    "xAcc": 0.0, "yAcc": 0.0, "zAcc": 0.0,
    "xGyro": 0.0, "yGyro": 0.0, "zGyro": 0.0,
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def _create_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def _get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None or email not in _users:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    name: str = ""          
class SensorWindow(BaseModel):
    window: list[list[float]]   
    latest_row: list[float] = []
    rows_collected: int = 0

class PredictionResponse(BaseModel):
    fall_detected: bool
    confidence: float
    risk_label: str
    risk_score: int
    timestamp: str

class SensorStatus(BaseModel):
    fall_detected: bool
    confidence: float
    risk_label: str
    risk_score: int
    timestamp: str
    sensor_connected: bool
    rows_collected: int
    xAcc: float
    yAcc: float
    zAcc: float
    xGyro: float
    yGyro: float
    zGyro: float

app = FastAPI(title="FallPredict API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/auth/signup", status_code=201)
async def signup(body: SignupRequest):
    email = body.email.strip().lower()
    if email in _users:
        raise HTTPException(400, detail="Email already registered")
    if len(body.password) < 6:
        raise HTTPException(400, detail="Password must be at least 6 characters")
    _users[email] = {
        "name": body.name.strip(),
        "hashed_password": hash_password(body.password),
    }
    return {"message": "Account created successfully"}


@app.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    email = body.email.strip().lower()
    user  = _users.get(email)
    if user is None or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return TokenResponse(
        access_token=_create_token(email),
        name=user["name"],       # FIX: include name
    )


@app.get("/auth/me")
async def me(current_user: str = Depends(_get_current_user)):
    return {"email": current_user, "name": _users[current_user]["name"]}


@app.post("/predict/fall", response_model=PredictionResponse)
async def predict_fall(
    body: SensorWindow,
    current_user: str = Depends(_get_current_user),
):
    if len(body.window) != WINDOW_SIZE:
        raise HTTPException(
            400,
            detail=f"Expected {WINDOW_SIZE} rows, got {len(body.window)}. "
                   f"Set WINDOW_SIZE={WINDOW_SIZE} in sensor_reader.py"
        )
    for row in body.window:
        if len(row) != N_FEATURES:
            raise HTTPException(400, detail=f"Each row needs {N_FEATURES} features")

    model, scaler = _load_model()
    if model is None or scaler is None:
        latest = body.latest_row if len(body.latest_row) == N_FEATURES else body.window[-1]
        _latest_status.update({
            "sensor_connected": True,
            "rows_collected": body.rows_collected,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "xAcc":  round(latest[0], 3),
            "yAcc":  round(latest[1], 3),
            "zAcc":  round(latest[2], 3),
            "xGyro": round(latest[3], 3),
            "yGyro": round(latest[4], 3),
            "zGyro": round(latest[5], 3),
        })
        raise HTTPException(503, detail="Model not loaded. Place fall_detection_model.keras and fall_detection_scaler.pkl next to backend.py")

    # ── Pre-process (same as training script) ──
    window_np   = np.array(body.window, dtype=np.float32)
    flat_scaled = scaler.transform(window_np.reshape(-1, N_FEATURES))

    if len(model.input_shape) == 3:
        X = flat_scaled.reshape(1, WINDOW_SIZE, N_FEATURES)
    else:
        X = flat_scaled.reshape(1, -1)

    raw_pred = model.predict(X, verbose=0)

    if raw_pred.shape[-1] == 1:
        prob_fall     = float(raw_pred[0][0])
        fall_detected = prob_fall >= 0.5
        confidence    = round(prob_fall if fall_detected else 1 - prob_fall, 4)
    else:
        pred_class    = int(np.argmax(raw_pred[0]))
        fall_detected = pred_class == 1
        confidence    = round(float(np.max(raw_pred[0])), 4)

    risk_score = int(confidence * 100)
    risk_label = "HIGH" if confidence > 0.75 else "MEDIUM" if confidence > 0.40 else "LOW"
    ts         = datetime.now(timezone.utc).isoformat()

    latest = body.latest_row if len(body.latest_row) == N_FEATURES else body.window[-1]

    _latest_status.update({
        "fall_detected":    fall_detected,
        "confidence":       round(confidence, 4),
        "risk_label":       risk_label,
        "risk_score":       risk_score,
        "timestamp":        ts,
        "sensor_connected": True,
        "rows_collected":   body.rows_collected,
        "xAcc":  round(latest[0], 3),
        "yAcc":  round(latest[1], 3),
        "zAcc":  round(latest[2], 3),
        "xGyro": round(latest[3], 3),
        "yGyro": round(latest[4], 3),
        "zGyro": round(latest[5], 3),
    })

    return PredictionResponse(
        fall_detected=fall_detected,
        confidence=round(confidence, 4),
        risk_label=risk_label,
        risk_score=risk_score,
        timestamp=ts,
    )


@app.get("/status/latest", response_model=SensorStatus)
async def latest_status():
    # Auto-detect disconnect
    try:
        last = datetime.fromisoformat(_latest_status["timestamp"])
        diff = (datetime.now(timezone.utc) - last).total_seconds()
        _latest_status["sensor_connected"] = diff < SENSOR_TIMEOUT_SEC
    except Exception:
        pass
    return _latest_status


@app.get("/health")
async def health():
    model, _ = _load_model()
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "window_size":  WINDOW_SIZE,
        "users_count":  len(_users),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

@app.get("/health/sensor")
async def health_sensor():
    """No-auth ping Flutter uses to check if backend is reachable."""
    return {"status": "ok", "sensor_connected": _latest_status["sensor_connected"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)