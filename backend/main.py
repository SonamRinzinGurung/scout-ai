# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import requests
# import json
# from typing import Optional

# app = FastAPI(title="Prospect Analyzer")

# origins = ["http://localhost:5173"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class PlayerStats(BaseModel):
#     name: str
#     position: str
#     age: Optional[int] = None
#     level: str  # "highschool", "college", "nba"
#     ppg: float
#     rpg: float
#     apg: float
#     fg_pct: float
#     three_pt_pct: float
#     team: Optional[str] = None  # school or NBA team


# def generate_prompt(player: PlayerStats):
#     descriptors = []

#     if player.ppg > 25:
#         descriptors.append("elite scorer")
#     elif player.ppg > 15:
#         descriptors.append("solid offensive option")
#     else:
#         descriptors.append("role player offensively")

#     if player.rpg > 10:
#         descriptors.append("dominant rebounder")

#     level_note = ""
#     if player.level == "highschool":
#         level_note = "Focus on how he might project in college and eventually NBA."
#     elif player.level == "college":
#         level_note = "Focus on how he might transition to the NBA."
#     else:
#         level_note = "Evaluate NBA potential and fit."

#     prompt = f"""
#     You are an NBA scout. Analyze this player:

#     Name: {player.name}
#     Position: {player.position}
#     Age: {player.age}
#     Team: {player.team}
#     Level: {player.level}
#     Stats: {player.ppg} PPG, {player.rpg} RPG, {player.apg} APG,
#            {player.fg_pct}% FG, {player.three_pt_pct}% 3PT
#     Descriptors: {', '.join(descriptors)}

#     {level_note}

#     Write a scouting report with strengths, weaknesses, and potential.
#     """
#     return prompt


# @app.post("/analyze")
# def analyze_player(player: PlayerStats):
#     prompt = generate_prompt(player)

#     def streaming_response():
#         with requests.post(
#             "http://127.0.0.1:11434/api/generate",
#             json={
#                 "model": "llama3.2",
#                 "prompt": prompt,
#                 "stream": True,
#                 "options": {"num_predict": 400},
#             },
#             stream=True,
#         ) as r:
#             for line in r.iter_lines():
#                 if not line:
#                     continue
#                 try:
#                     obj = json.loads(line.decode("utf-8"))
#                     # Forward the JSON exactly like Ollama sends
#                     yield json.dumps(obj) + "\n"
#                 except json.JSONDecodeError:
#                     # In case Ollama sends partial chunks
#                     continue

#     return StreamingResponse(streaming_response(), media_type="application/json")


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import os
from typing import Optional
from ml.models.nba_draft_predictor import NBADraftPredictor

app = FastAPI(title="Prospect Analyzer")

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NBA Draft Predictor
try:
    # Update with your actual filename
    MODEL_PATH = "ml/models/nba_draft_ensemble_model_20250925.pkl"
    # Update with your actual filename
    PREPROCESSOR_PATH = "ml/models/nba_draft_preprocessors_20250925.pkl"

    # Check if files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        draft_predictor = NBADraftPredictor(MODEL_PATH, PREPROCESSOR_PATH)
        print("✅ NBA Draft Predictor loaded successfully!")
    else:
        draft_predictor = None
        print("❌ Model files not found. Draft prediction will be unavailable.")
except Exception as e:
    draft_predictor = None
    print(f"❌ Failed to load NBA Draft Predictor: {e}")


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


class CollegePlayerStats(BaseModel):
    """Extended model for college players with advanced stats"""
    name: str
    position: str  # Maps to 'role_position'
    age: Optional[int] = None
    level: str = "college"
    team: Optional[str] = None  # Maps to 'conf'
    year: Optional[str] = None  # Maps to 'yr' (Fr, So, Jr, Sr)

    # Basic stats
    ppg: float  # Maps to 'pts'
    rpg: float  # Maps to 'treb'
    apg: float  # Maps to 'ast'
    fg_pct: float  # Used to calculate efficiency metrics
    three_pt_pct: float  # Maps to 'TP_per'
    ft_pct: Optional[float] = None  # Maps to 'FT_per'

    # Advanced stats (optional - will use defaults if not provided)
    minutes_per_game: Optional[float] = None  # Maps to 'Min_per'
    usage_rate: Optional[float] = None  # Maps to 'usg'
    games_played: Optional[int] = None  # Maps to 'GP'
    true_shooting_pct: Optional[float] = None  # Maps to 'TS_per'
    box_plus_minus: Optional[float] = None  # Maps to 'bpm'
    steal_pct: Optional[float] = None  # Maps to 'stl_per'
    block_pct: Optional[float] = None  # Maps to 'blk_per'


def convert_to_model_format(player: CollegePlayerStats) -> dict:
    """Convert PlayerStats to the format expected by the ML model"""

    # Map position abbreviations
    position_map = {
        'PG': 'PG', 'Point Guard': 'PG',
        'SG': 'SG', 'Shooting Guard': 'SG', 'Guard': 'SG',
        'SF': 'SF', 'Small Forward': 'SF', 'Forward': 'SF',
        'PF': 'PF', 'Power Forward': 'PF',
        'C': 'C', 'Center': 'C'
    }

    # Map conference (simplified)
    conf_map = {
        'Duke': 'ACC', 'North Carolina': 'ACC', 'Virginia': 'ACC',
        'Kansas': 'Big 12', 'Texas': 'Big 12',
        'Michigan': 'Big Ten', 'Ohio State': 'Big Ten',
        'Kentucky': 'SEC', 'Alabama': 'SEC',
        'UCLA': 'Pac-12', 'Arizona': 'Pac-12',
        'Villanova': 'Big East', 'Georgetown': 'Big East'
    }

    # Create model input with available data and reasonable defaults
    model_data = {
        'player_name': player.name,
        # Default to 'Other' if not found
        'conf': conf_map.get(player.team, 'Other'),
        'yr': player.year or 'Jr',  # Default to Junior
        'role_position': position_map.get(player.position, player.position),

        # Basic stats
        'pts': player.ppg,
        'treb': player.rpg,
        'ast': player.apg,
        'TP_per': player.three_pt_pct / 100.0,  # Convert percentage to decimal
        'FT_per': (player.ft_pct or 75.0) / 100.0,  # Default 75% FT

        # Playing time and usage (use defaults if not provided)
        'Min_per': player.minutes_per_game or (30.0 if player.ppg > 15 else 20.0),
        'usg': player.usage_rate or (25.0 if player.ppg > 20 else 20.0),
        'GP': player.games_played or 30,

        # Efficiency metrics (calculated or default)
        # Effective FG%
        'eFG': (player.fg_pct + 0.5 * player.three_pt_pct) / 100.0,
        'TS_per': player.true_shooting_pct / 100.0 if player.true_shooting_pct else (player.fg_pct / 100.0 + 0.05),
        # Estimated 2P%
        'twoP_per': max(0.4, (player.fg_pct - 0.3 * player.three_pt_pct) / 100.0),
        # Estimated assist/turnover ratio
        'ast/tov': max(0.8, player.apg / 2.5),

        # Advanced metrics (use defaults based on production)
        'Ortg': 100 + (player.ppg - 12) * 2,  # Estimated offensive rating
        'bpm': player.box_plus_minus or (player.ppg + player.rpg + player.apg - 20) / 3,
        'obpm': (player.box_plus_minus or 0) * 0.6 if player.box_plus_minus else (player.ppg - 12) / 4,
        'dbpm': (player.box_plus_minus or 0) * 0.4 if player.box_plus_minus else (player.rpg - 5) / 5,
        'adjoe': 105 + (player.ppg - 15),
        'drtg': max(90, 105 - (player.rpg - 3) * 2),
        'adrtg': max(90, 105 - (player.rpg - 3) * 2),

        # Percentage stats (estimates)
        'ORB_per': max(2.0, (player.rpg * 0.3)),
        'DRB_per': max(5.0, (player.rpg * 0.7)),
        'AST_per': max(5.0, player.apg * 3),
        'TO_per': max(8.0, 20 - player.apg),
        'blk_per': player.block_pct or (2.0 if player.position in ['C', 'PF'] else 0.5),
        'stl_per': player.steal_pct or 1.5,
        'ftr': 0.3 + (player.ppg - 10) * 0.02,  # Estimated free throw rate

        # Shooting breakdown (estimates)
        'rimmade/(rimmade+rimmiss)': max(0.5, min(0.85, 0.6 + (player.fg_pct - 45) / 100)),
        'midmade/(midmade+midmiss)': max(0.3, min(0.6, player.fg_pct / 100 - 0.1)),
        'dunksmade/(dunksmade+dunksmiss)': 0.9 if player.position in ['C', 'PF'] else 0.7,

        # Additional counting stats
        'stl': max(0.5, player.apg * 0.3),  # Estimated steals
        # Estimated blocks
        'blk': 1.0 if player.position in ['C', 'PF'] else 0.3,
    }

    return model_data


def generate_enhanced_prompt(player: PlayerStats, draft_prediction: dict = None):
    """Generate prompt enhanced with ML model predictions"""

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

    # Add draft prediction context if available
    draft_context = ""
    if draft_prediction and 'error' not in draft_prediction:
        prob = draft_prediction['draft_probability']
        prediction = draft_prediction['prediction']
        confidence = draft_prediction['confidence']

        draft_context = f"""
        
        ML MODEL ANALYSIS:
        Draft Probability: {prob:.1%}
        Prediction: {prediction}
        Confidence Level: {confidence}
        
        Based on historical data from 2009-2021 college players, this model suggests this player has a {prob:.1%} chance of being drafted.
        """

    prompt = f"""
    You are an NBA scout with access to advanced analytics. Analyze this player:

    Name: {player.name}
    Position: {player.position}
    Age: {player.age}
    Team: {player.team}
    Level: {player.level}
    Stats: {player.ppg} PPG, {player.rpg} RPG, {player.apg} APG,
           {player.fg_pct}% FG, {player.three_pt_pct}% 3PT
    Descriptors: {', '.join(descriptors)}
    {draft_context}

    {level_note}

    Write a comprehensive scouting report that includes:
    1. Strengths and weaknesses
    2. NBA potential and projection
    3. Draft outlook (if applicable)
    
    Consider both the traditional stats and the ML model's assessment in your analysis.
    """
    return prompt


@app.post("/analyze")
def analyze_player(player: PlayerStats):
    """Analyze player with optional ML draft prediction"""

    draft_prediction = None

    # Try to get draft prediction for college players
    if player.level == "college" and draft_predictor:
        try:
            # Convert basic PlayerStats to extended format for ML model
            college_player = CollegePlayerStats(
                name=player.name,
                position=player.position,
                age=player.age,
                team=player.team,
                ppg=player.ppg,
                rpg=player.rpg,
                apg=player.apg,
                fg_pct=player.fg_pct,
                three_pt_pct=player.three_pt_pct
            )

            model_data = convert_to_model_format(college_player)
            draft_prediction = draft_predictor.predict_draft_probability(
                model_data)
            print(f"✅ Draft prediction successful: {draft_prediction}")
        except Exception as e:
            print(f"⚠️ Draft prediction failed: {e}")
            draft_prediction = {'error': str(e)}

    # Generate enhanced prompt
    prompt = generate_enhanced_prompt(player, draft_prediction)

    def streaming_response():
        with requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": True,
                # Increased for more detailed analysis
                "options": {"num_predict": 800},
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


@app.post("/draft-prediction")
def predict_draft(player: CollegePlayerStats):
    """Dedicated endpoint for draft prediction"""
    if not draft_predictor:
        raise HTTPException(
            status_code=503, detail="Draft predictor not available")

    try:
        model_data = convert_to_model_format(player)
        prediction = draft_predictor.predict_draft_probability(model_data)
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "draft_predictor_available": draft_predictor is not None,
        "ollama_connection": "assumed_healthy"  # Could add actual check
    }
