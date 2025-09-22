from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
from typing import Optional

app = FastAPI(title="Prospect Analyzer")

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerStats(BaseModel):
    name: str
    position: str
    age: Optional[int] = None
    level: str  # "highschool", "college", "nba"
    ppg: float
    rpg: float
    apg: float
    fg_pct: float
    three_pt_pct: float
    team: Optional[str] = None  # school or NBA team


def generate_prompt(player: PlayerStats):
    descriptors = []

    if player.ppg > 25:
        descriptors.append("elite scorer")
    elif player.ppg > 15:
        descriptors.append("solid offensive option")
    else:
        descriptors.append("role player offensively")

    if player.rpg > 10:
        descriptors.append("dominant rebounder")

    level_note = ""
    if player.level == "highschool":
        level_note = "Focus on how he might project in college and eventually NBA."
    elif player.level == "college":
        level_note = "Focus on how he might transition to the NBA."
    else:
        level_note = "Evaluate NBA potential and fit."

    prompt = f"""
    You are an NBA scout. Analyze this player:

    Name: {player.name}
    Position: {player.position}
    Age: {player.age}
    Team: {player.team}
    Level: {player.level}
    Stats: {player.ppg} PPG, {player.rpg} RPG, {player.apg} APG,
           {player.fg_pct}% FG, {player.three_pt_pct}% 3PT
    Descriptors: {', '.join(descriptors)}

    {level_note}

    Write a scouting report with strengths, weaknesses, and potential.
    """
    return prompt


@app.post("/analyze")
def analyze_player(player: PlayerStats):
    prompt = generate_prompt(player)

    def streaming_response():
        with requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": 400},
            },
            stream=True,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    # Forward the JSON exactly like Ollama sends
                    yield json.dumps(obj) + "\n"
                except json.JSONDecodeError:
                    # In case Ollama sends partial chunks
                    continue

    return StreamingResponse(streaming_response(), media_type="application/json")
