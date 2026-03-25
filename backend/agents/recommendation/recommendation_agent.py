"""
Recommendation Agent – Top-level coordinator for the flood recommendation pipeline.

Execution pipeline:
  1. UserProfiler          : Infer user type, vulnerability, and accessibility needs
  2. LocationContextBuilder: Enrich geographic and infrastructure context
  3. RecommendationEngine  : Generate LLM + rule-based recommendations
  4. Safety message         : Short SMS-friendly alert text
  5. Authority brief        : Formal brief for officials (if authority user)
  6. Store artifacts        : All outputs stored in session

This agent is registered in the ToolRegistry under "recommendation" and called
by the Orchestrator after the PredictionAgent completes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from agents.recommendation.recommendation_schemas import (
    LocationContext as LocationContextModel,
    Recommendation,
    RecommendationResult,
    ResourceAllocation,
    UrgencyLevel,
    UserProfile,
    UserType,
)
from agents.recommendation.user_profiler import UserProfiler
from agents.recommendation.location_context import LocationContextBuilder
from agents.recommendation.recommendation_engine import RecommendationEngine
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


class RecommendationAgent:
    """
    Orchestrates the full recommendation generation pipeline.

    All sub-components are injected and stateless — the agent can be reused
    across multiple sessions.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini   = gemini_service or get_gemini_service()
        self._profiler = UserProfiler(self._gemini)
        self._location = LocationContextBuilder(self._gemini)
        self._engine   = RecommendationEngine(self._gemini)

    def _get_regional_language(self, state: str, location: str = "") -> str:
        """Maps Indian states or major cities to their primary regional language."""
        # Check state first, then location name for common cities
        s = (state or "").lower()
        loc = (location or "").lower()
        
        if "tamil" in s or "chennai" in loc: return "Tamil"
        if "kerala" in s or "kochi" in loc or "trivandrum" in loc: return "Malayalam"
        if "karnataka" in s or "bangalore" in loc or "bengaluru" in loc: return "Kannada"
        if "andhra" in s or "telangana" in s or "hyderabad" in s or "amaravati" in loc: return "Telugu"
        if "bengal" in s or "kolkata" in loc: return "Bengali"
        if "maharashtra" in s or "mumbai" in loc or "pune" in loc: return "Marathi"
        if "gujarat" in s or "ahmedabad" in loc or "surat" in loc: return "Gujarati"
        if "odisha" in s or "bhubaneswar" in loc: return "Odia"
        if "punjab" in s or "amritsar" in loc or "ludhiana" in loc: return "Punjabi"
        if "assam" in s or "guwahati" in loc: return "Assamese"
        
        # Hindi-speaking belt
        if any(x in s for x in ("uttar", "bihar", "madhya", "rajasthan", "haryana", "delhi", "himachal", "uttarakhand", "jharkhand", "chhattisgarh")):
            return "Hindi"
        if any(x in loc for x in ("delhi", "patna", "lucknow", "jaipur", "bhopal", "ranchi", "raipur")):
            return "Hindi"
            
        return "Hindi"  # Default fallback for India context

    # ── Public interface (called by Orchestrator) ─────────────────────────

    async def run(
        self,
        session: Session,
        **kwargs,
    ) -> RecommendationResult:
        """
        Main agent entry point — uses a single merged LLM call for all outputs.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[RecommendationAgent] Starting for session {session_id}")

        try:
            # ── 1. Retrieve prediction results ────────────────────────────
            prediction = session.get_artifact("ensemble_prediction")
            if not prediction or not isinstance(prediction, dict):
                return RecommendationResult(
                    session_id=session_id,
                    status="failed",
                    errors=["No prediction results found. Run PredictionAgent first."],
                )

            risk_level = prediction.get("risk_level", "MEDIUM")
            flood_prob = prediction.get("flood_probability", 0.5)
            confidence = prediction.get("confidence", 0.5)

            # ── 2. Gather context for the merged prompt ───────────────────
            location_name = context.get("location", "Unknown")
            state         = context.get("state", "")
            country       = context.get("country", "India")
            user_type     = context.get("user_type", "general_public")
            has_elderly   = context.get("has_elderly", False)
            has_children  = context.get("has_children", False)
            has_disability = context.get("has_disability", False)
            vehicle_access = context.get("vehicle_access", True)

            df = session.get_artifact("processed_dataset")
            env_data = self._extract_env_data(df, prediction)
            env_str  = ", ".join(f"{k}={v}" for k, v in env_data.items() if k not in ("risk_level",))

            # ── 3. Single merged LLM call ─────────────────────────────────
            regional_lang = self._get_regional_language(state, location_name)
            logger.info(f"[RecommendationAgent] Detected regional hint: {regional_lang} (state={state}, loc={location_name})")
            
            prompt = f"""You are a flood disaster management AI assistant for India.
Generate a complete recommendation report in ONE JSON response.

## Flood Context
- Location: {location_name}, {state}, {country}
- Risk Level: {risk_level}
- Flood Probability: {flood_prob*100:.0f}%
- Confidence: {confidence*100:.0f}%
- Environmental readings: {env_str or 'N/A'}
- Preferred Local Language: {regional_lang}

## Instructions
1. For EVERY recommendation, you MUST provide BOTH English and the local language spoken in {location_name} (e.g. {regional_lang} for this area).
2. Use the local language for all `_regional` fields in the schema. Do NOT leave them empty.
3. Be specific to the geography of {location_name}.
4. Output MUST be valid JSON (no prose, no markdown fences).

## Output Schema
{{
  "recommendations": [
    {{
      "category": "<safety|evacuation|infrastructure|resource|health|communication>",
      "urgency": "<emergency|warning|advisory|informational>",
      "title": "<short action title in English>",
      "title_regional": "<SAME title translated into the local language of {location_name}>",
      "description": "<1-2 sentence description in English>",
      "description_regional": "<SAME description translated into the local language of {location_name}>",
      "action_steps": ["<step 1 in English>", "<step 2>", "<step 3>"],
      "action_steps_regional": ["<step 1 in local language>", "<step 2>", "<step 3>"],
      "priority": <integer 1–10>
    }}
  ],
  "safety_message": "<SMS-friendly safety message under 160 chars>",
  "summary": "<2-3 sentence overall summary for a {user_type} user>",
  "emergency_contact": "112",
  "risk_factors": ["<factor 1>", "<factor 2>"],
  "resources": [
    {{
      "resource_type": "<personnel|equipment|shelter|medical|food_water>",
      "quantity": <integer>,
      "deploy_to": "{location_name}",
      "urgency": "<emergency|warning|advisory|informational>",
      "rationale": "<why needed>"
    }}
  ]
}}"""

            try:
                raw = await self._gemini.generate_json(prompt)
            except Exception as e:
                logger.warning(f"[RecommendationAgent] Gemini generation error: {e}")
                raw = None

            if not raw or not isinstance(raw, dict) or not raw.get("recommendations"):
                warnings.append("Gemini failed — attempting Ollama (Llama3) local fallback...")
                raw = await self._try_ollama_fallback(prompt)
                
            if not raw or not isinstance(raw, dict) or not raw.get("recommendations"):
                warnings.append("Ollama failed — using emergency static fallback.")
                raw = self._get_emergency_fallback(risk_level, location_name, state)
            
            if not isinstance(raw, dict):
                raw = {"recommendations": [], "resources": []}
            
            logger.info(f"[RecommendationAgent] Raw LLM response keys: {list(raw.keys())}")
            
            # ── 4. Parse LLM output ───────────────────────────────────────
            from agents.recommendation.recommendation_schemas import (
                Recommendation, RecommendationCategory, ResourceAllocation,
            )

            recs: List[Recommendation] = []
            for item in raw.get("recommendations", []):
                try:
                    recs.append(Recommendation(
                        id=item.get("id", len(recs) + 1),
                        category=RecommendationCategory(item.get("category", "safety")),
                        urgency=UrgencyLevel(item.get("urgency", "advisory")),
                        title=item.get("title", "Stay safe"),
                        title_regional=item.get("title_regional"),
                        description=item.get("description", ""),
                        description_regional=item.get("description_regional"),
                        action_steps=item.get("action_steps", []),
                        action_steps_regional=item.get("action_steps_regional", []),
                        priority=item.get("priority", 5),
                    ))
                except Exception:
                    pass

            emergency_number = raw.get("emergency_contact", "112")

            resources: List[ResourceAllocation] = []
            for item in raw.get("resources", []):
                try:
                    resources.append(ResourceAllocation(
                        resource_type=item.get("resource_type", "personnel"),
                        quantity=item.get("quantity", 1),
                        deploy_to=item.get("deploy_to", location_name),
                        urgency=UrgencyLevel(item.get("urgency", "advisory")),
                        rationale=item.get("rationale", "Emergency deployment"),
                    ))
                except Exception:
                    pass

            safety_msg = raw.get("safety_message") or (
                f"⚠️ {risk_level} flood risk in {location_name}. "
                f"Flood probability: {flood_prob*100:.0f}%. Call 112 for emergencies."
            )
            summary = raw.get("summary") or self._fallback_summary_text(
                risk_level, flood_prob, recs, location_name, emergency_number
            )

            if not recs:
                warnings.append("No recommendations generated — using defaults.")
                recs = self._default_recommendations_simple(risk_level, location_name, emergency_number)

            loc_ctx = LocationContextModel(
                location_name=location_name,
                state=state,
                country=country,
                emergency_number=emergency_number,
                risk_factors=raw.get("risk_factors", []),
            )

            profile = UserProfile(
                user_type=UserType(user_type) if user_type in UserType._value2member_map_ else UserType.PUBLIC,
                has_elderly=has_elderly,
                has_children=has_children,
                has_disability=has_disability,
                vehicle_access=vehicle_access,
            )

            urgency = self._overall_urgency(recs)

            session.store_artifact("recommendations", [r.model_dump() for r in recs])
            session.store_artifact("resource_plan", [r.model_dump() for r in resources])
            session.store_artifact("safety_message", safety_msg)
            session.store_artifact("recommendation_summary", summary)

            logger.info(
                f"[RecommendationAgent] Complete: {len(recs)} recs (1 LLM call) | "
                f"urgency={urgency} | user_type={profile.user_type}"
            )

            return RecommendationResult(
                session_id=session_id,
                status="success" if not warnings else "partial",
                risk_level=risk_level,
                urgency=urgency,
                recommendations=recs,
                resource_plan=resources,
                summary=summary,
                safety_message=safety_msg,
                user_profile=profile,
                location_context=loc_ctx,
                warnings=warnings,
                errors=errors,
            )

        except Exception as exc:
            logger.exception(f"[RecommendationAgent] Unhandled error: {exc}")
            return RecommendationResult(
                session_id=session_id,
                status="failed",
                errors=[str(exc)],
            )

    @staticmethod
    def _fallback_summary_text(
        risk_level: str, flood_prob: float,
        recs: List, location: str, emergency_number: str,
    ) -> str:
        return (
            f"Flood risk is {risk_level} for {location} "
            f"({flood_prob*100:.0f}% probability). "
            f"{len(recs)} recommendations generated. "
            f"Emergency: {emergency_number}."
        )

    @staticmethod
    def _default_recommendations_simple(
        risk_level: str, location_name: str, emergency_number: str,
    ) -> List:
        from agents.recommendation.recommendation_schemas import (
            Recommendation, RecommendationCategory,
        )
        return [
            Recommendation(
                id=1,
                category=RecommendationCategory.SAFETY,
                urgency=UrgencyLevel.ADVISORY,
                title="Stay Informed About Flood Risk",
                description=(
                    f"Flood risk is {risk_level} in {location_name or 'your area'}. "
                    "Monitor local weather updates and disaster management advisories."
                ),
                action_steps=[
                    "Check local weather forecast regularly",
                    f"Call {emergency_number} if you need help",
                    "Keep emergency supplies ready",
                ],
                priority=1,
            ),
        ]

    @staticmethod
    def _extract_env_data(
        df:         Optional[pd.DataFrame],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        env: Dict[str, Any] = {}
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            env_cols = [
                "rainfall_mm", "water_level_m", "discharge_m3s",
                "soil_moisture_pct", "elevation_m", "temperature_c",
                "humidity_pct", "wind_speed_kmh", "rolling_7d_rainfall",
                "rolling_14d_rainfall", "amc", "flood_risk_proxy",
            ]
            for col in env_cols:
                if col in df.columns:
                    val = last.get(col)
                    if pd.notna(val):
                        env[col] = round(float(val), 2)
        env["flood_probability"] = prediction.get("flood_probability", 0.5)
        env["risk_level"]        = prediction.get("risk_level", "MEDIUM")
        return env

    @staticmethod
    def _overall_urgency(recs: List[Recommendation]) -> UrgencyLevel:
        urgency_order = [
            UrgencyLevel.EMERGENCY,
            UrgencyLevel.WARNING,
            UrgencyLevel.ADVISORY,
            UrgencyLevel.INFORMATIONAL,
        ]
        for u in urgency_order:
            if any(r.urgency == u for r in recs):
                return u
        return UrgencyLevel.INFORMATIONAL

    async def _generate_summary(
        self,
        risk_level: str,
        flood_prob: float,
        recs:       List[Recommendation],
        profile:    UserProfile,
        location:   LocationContextModel,
    ) -> str:
        try:
            rec_titles = [f"{r.priority}. {r.title}" for r in recs[:5]]
            prompt = f"""Summarize these flood recommendations in 2-3 sentences for a {profile.user_type.value} user.
Location: {location.location_name} | Risk: {risk_level} | Probability: {flood_prob*100:.0f}%

Top recommendations:
{chr(10).join(rec_titles)}

Be concise, direct, and action-oriented. No headers or bullets."""
            summary = await self._gemini.generate(prompt, use_fast_model=True)
            return summary.strip() if summary else self._fallback_summary(
                risk_level, flood_prob, recs, location
            )
        except Exception:
            return self._fallback_summary(risk_level, flood_prob, recs, location)

    @staticmethod
    def _fallback_summary(
        risk_level: str,
        flood_prob: float,
        recs:       List[Recommendation],
        location:   LocationContextModel,
    ) -> str:
        loc = location.location_name or "your area"
        return (
            f"Flood risk is {risk_level} for {loc} "
            f"({flood_prob*100:.0f}% probability). "
            f"{len(recs)} recommendations have been generated. "
            f"Emergency contact: {location.emergency_number}."
        )

    async def _try_ollama_fallback(self, prompt: str) -> Dict[str, Any]:
        """Attempts to generate recommendations using local Ollama (Llama3) via httpx."""
        import httpx
        try:
            logger.info("[RecommendationAgent] Attempting local Ollama fallback (llama3:8b)...")
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=240.0)
                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "{}")
                    logger.info(f"[RecommendationAgent] Ollama produced {len(response_text)} chars.")
                    import json
                    return json.loads(response_text)
                else:
                    logger.warning(f"[RecommendationAgent] Ollama returned status {resp.status_code}")
        except Exception as e:
            logger.warning(f"[RecommendationAgent] Ollama fallback failed: {str(e)}")
        return {}

    def _get_emergency_fallback(self, risk_level: str, location: str, state: str = "") -> Dict[str, Any]:
        """Provides hardcoded recommendations with regional support for common Indian languages."""
        is_high = risk_level in ["HIGH", "CRITICAL", "RiskLevel.CRITICAL", "RiskLevel.HIGH"]
        lang_name = self._get_regional_language(state, location)
        
        translations = {
            "Tamil": {
                "t1": "மேட்டுப்பகுதிக்கு வெளியேறவும்",
                "d1": "தாழ்வான பகுதிகளில் வசிப்பவர்கள் உடனடியாக வெளியேற பரிந்துரைக்கப்படுகிறார்கள்.",
                "s1": ["அத்தியாவசியப் பொருட்களை எடுத்துச் செல்லுங்கள்", "நியமிக்கப்பட்ட தப்பிக்கும் வழிகளைப் பின்பற்றுங்கள்", "உயர்ந்த இடத்திற்குச் செல்லுங்கள்"],
                "t2": "அவசரக்கால பெட்டியை தயார் செய்யவும்",
                "d2": "குறைந்தது 72 மணிநேரத்திற்கு தண்ணீர் மற்றும் மருத்துவ பொருட்கள் இருப்பதை உறுதி செய்யவும்.",
                "s2": ["லைட்டுகளை சரிபார்க்கவும்", "பவர் பேங்குகளை சார்ஜ் செய்யவும்", "முக்கிய ஆவணங்களை பாதுகாக்கவும்"],
                "t3": "சொத்துக்களை பாதுகாக்கவும்",
                "d3": "விலை உயர்ந்த பொருட்களை மேல் தளத்திற்கு மாற்றவும், மின்சாரத்தை அணைக்கவும்.",
                "s3": ["மணல் மூட்டைகளைப் பயன்படுத்தவும்", "மின் சாதனங்களை அன்ப்ளக் செய்யவும்", "வடிகால்களை சுத்தம் செய்யவும்"]
            },
            "Hindi": {
                "t1": "ऊंचे स्थानों पर चले जाएं",
                "d1": "निचले इलाकों या नदियों के पास रहने वाले लोगों को तुरंत सुरक्षित स्थानों पर जाने की सलाह दी जाती है।",
                "s1": ["जरूरी सामान साथ रखें", "निर्धारित सुरक्षित रास्तों का पालन करें", "ऊंचे स्थानों की ओर बढ़ें"],
                "t2": "आपातकालीन किट तैयार रखें",
                "d2": "72 घंटों के लिए पानी, भोजन और दवाओं की व्यवस्था सुनिश्चित करें।",
                "s2": ["टॉर्च की जाँच करें", "पावर बैंक चार्ज करें", "महत्वपूर्ण दस्तावेजों को सुरक्षित रखें"],
                "t3": "संपत्ति की सुरक्षा",
                "d3": "कीमती सामान ऊपरी मंजिलों पर ले जाएं और बिजली बंद कर दें।",
                "s3": ["रेत की बोरियों का उपयोग करें", "बिजली के उपकरणों को अनप्लग करें", "नालियों को साफ रखें"]
            },
            "Malayalam": {
                "t1": "താഴ്ന്ന പ്രദേശങ്ങളിൽ നിന്ന് ഒഴിഞ്ഞുമാറുക",
                "d1": "വെള്ളപ്പൊക്ക സാധ്യതയുള്ള പ്രദേശങ്ങളിൽ താമസിക്കുന്നവർ ഉടൻ സുരക്ഷിത സ്ഥാനങ്ങളിലേക്ക് മാറണം.",
                "s1": ["അത്യാവശ്യ സാധനങ്ങൾ കരുതുക", "നിർദ്ദിഷ്ട പാതകൾ പിന്തുടരുക", "ഉയർന്ന സ്ഥലത്തേക്ക് നീങ്ങുക"],
                "t2": "അടിയന്തര കിറ്റ് തയ്യാറാക്കുക",
                "d2": "72 മണിക്കൂറത്തേക്കുള്ള വെള്ളം, ഭക്ഷണം, മരുന്നുകൾ എന്നിവ ഉറപ്പാക്കുക.",
                "s2": ["ടോർച്ച് പരിശോധിക്കുക", "പവർ ബാങ്ക് ചാർജ് ചെയ്യുക", "പ്രധാന രേഖകൾ സുരക്ഷിതമാക്കുക"],
                "t3": "സ്വത്ത് സംരക്ഷിക്കുക",
                "d3": "വിലപിടിപ്പുള്ള വസ്തുക്കൾ മുകളിലത്തെ നിലയിലേക്ക് മാറ്റുക, വൈദ്യുതി/ഗ്യാസ് ഓഫ് ചെയ്യുക.",
                "s3": ["മണൽ ചാക്കുകൾ ഉപയോഗിക്കുക", "വൈദ്യുതി ബന്ധം വേർപെടുത്തുക", "ഓടകൾ വൃത്തിയാക്കുക"]
            },
            "Marathi": {
                "t1": "सखल भागातून बाहेर पडा",
                "d1": "नदीकाठच्या आणि सखल भागातील नागरिकांनी तातडीने सुरक्षित स्थळी स्थलांतर करावे.",
                "s1": ["जीवनावश्यक वस्तू सोबत घ्या", "ठरवून दिलेल्या मार्गाचा वापर करा", "उंचावर जा"],
                "t2": "आणीबाणीची किट तयार ठेवा",
                "d2": "किमान ७२ तास पुरेल इतका अन्न, पाणी आणि औषधांचा साठा करा.",
                "s2": ["बॅटरी तपासा", "पॉवर बँक चार्ज करा", "कागदपत्रे सुरक्षित ठेवा"],
                "t3": "मालमत्तेचे रक्षण करा",
                "d3": "किमती वस्तू वरच्या मजल्यावर हलवा आणि वीज/गॅस बंद करा.",
                "s3": ["वाळूच्या पोत्यांचा वापर करा", "वीज बंद करा", "नाले स्वच्छ करा"]
            },
            "Bengali": {
                "t1": "নিচু এলাকা থেকে সরে যান",
                "d1": "বন্যাকবলিত বা নদী তীরের বাসিন্দাদের অবিলম্বে নিরাপদ স্থানে চলে যাওয়ার পরামর্শ দেওয়া হচ্ছে।",
                "s1": ["প্রয়োজনীয় জিনিস গুছিয়ে নিন", "নির্দিষ্ট নিরাপদ পথ ব্যবহার করুন", "উঁচু স্থানে আশ্রয় নিন"],
                "t2": "জরুরি সরঞ্জাম প্রস্তুত রাখুন",
                "d2": "কমপক্ষে ৭২ ঘণ্টার জন্য পানীয় জল, খাবার এবং ওষুধ মজুদ রাখুন।",
                "s2": ["টর্চ পরীক্ষা করুন", "পাওয়ার ব্যাঙ্ক চার্জ করুন", "গুরুত্বপূর্ণ নথি সুরক্ষিত রাখুন"],
                "t3": "সম্পদ রক্ষা করুন",
                "d3": "মূল্যবান জিনিস ওপরের তলায় সরিয়ে দিন এবং বিদ্যুৎ/গ্যাস সংযোগ বিচ্ছিন্ন করুন।",
                "s3": ["বালির বস্তা ব্যবহার করুন", "বৈদ্যুতিক যন্ত্র খুলে দিন", "ড্রেনেজ পরিষ্কার করুন"]
            }
        }
        
        lang_data = translations.get(lang_name, {})

        return {
            "status": "partial",
            "summary": f"Emergency safety guidance for {location} due to {risk_level} flood risk.",
            "recommendations": [
                {
                    "id": 1,
                    "category": "safety",
                    "urgency": "emergency" if is_high else "advisory",
                    "title": "Evacuate if in Low-Lying Areas",
                    "title_regional": lang_data.get("t1"),
                    "description": "Immediate evacuation is recommended for residents in floodplains or near riverbanks.",
                    "description_regional": lang_data.get("d1"),
                    "action_steps": ["Pack essentials", "Follow designated escape routes", "Move to higher ground"],
                    "action_steps_regional": lang_data.get("s1", [])
                },
                {
                    "id": 2,
                    "category": "safety",
                    "urgency": "warning",
                    "title": "Prepare Emergency Kit",
                    "title_regional": lang_data.get("t2"),
                    "description": "Ensure you have water, non-perishable food, and medical supplies for at least 72 hours.",
                    "description_regional": lang_data.get("d2"),
                    "action_steps": ["Check flashlights", "Charge power banks", "Secure important documents"],
                    "action_steps_regional": lang_data.get("s2", [])
                },
                {
                    "id": 3,
                    "category": "infrastructure",
                    "urgency": "advisory",
                    "title": "Protect Property",
                    "title_regional": lang_data.get("t3"),
                    "description": "Move valuable items to upper floors and turn off electricity/gas if water enters.",
                    "description_regional": lang_data.get("d3"),
                    "action_steps": ["Use sandbags if available", "Unplug appliances", "Clear drainage holes"],
                    "action_steps_regional": lang_data.get("s3", [])
                }
            ],
            "resources": [
                {
                    "resource_type": "rescue_boats",
                    "quantity": 5,
                    "deploy_to": location,
                    "urgency": "emergency",
                    "rationale": "Required for potential rescue operations in urban flooding."
                }
            ]
        }
