FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY scenario_templates.py .
COPY scenario_pipeline.py .
COPY final_models.pkl .
COPY scenario_models.pkl .
COPY growth_caps.json .
COPY scenario_status.json .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0" , "--port", "8000"]