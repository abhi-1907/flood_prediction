# Intelligent Flood Prediction & Support System

An agentic AI system for flood prediction, simulation, recommendations, and alerting.

## Tech Stack
- **Backend**: FastAPI (Python), Gemini API (LLM), XGBoost, RandomForest, LSTM
- **Frontend**: React + Vite, Leaflet.js (maps)

## Agents
1. **Orchestration Agent** – Central LLM planner and router
2. **Data Ingestion Agent** – Multi-source data collection and structuring
3. **Preprocessing Agent** – Intelligent data cleaning and transformation
4. **Prediction Agent** – Multi-model flood risk prediction
5. **Recommendation Agent** – Location/user-aware LLM recommendations
6. **Simulation Agent** – Flood zone mapping and GeoJSON rendering
7. **Alerting Agent** – Email/SMS/push alerting pipeline

## Setup
See `.env.example` for required environment variables.

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
