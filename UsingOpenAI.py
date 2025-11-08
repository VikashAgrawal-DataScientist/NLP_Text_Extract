from typing import List, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

class UserProfile(BaseModel):
    stress_triggers: List[str] = Field(default_factory=list, description="List of stress triggers mentioned")
    positive_habits: List[str] = Field(default_factory=list, description="List of positive habits and activities")
    emotional_trends: List[str] = Field(default_factory=list, description="List of emotional words detected")
    sleep_mention_count: int = Field(default=0, description="Count of sleep-related mentions")
    number_of_children: int = Field(default=0, description="Number of children mentioned")
    marital_status: str = Field(default="unknown", description="Marital status inferred from context")
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(default="NEUTRAL",description="Sentiment of the text")

class ExtractionResult(BaseModel):
    profile: UserProfile
    confidence: float = Field(..., description="Overall confidence score for the extraction")

def create_extraction_prompt(text: str) -> str:
    return f"""
Analyze the following text and extract structured information about the user's wellness profile.

TEXT TO ANALYZE:
{text}

EXTRACTION INSTRUCTIONS:

1. STRESS TRIGGERS: Identify specific sources of stress or anxiety mentioned (e.gl, 'work', 'deadline', 'pressure', 'stress', 'anxious', 'worried', 'overwhelmed',
    'busy', 'rush', 'urgent', 'behind', 'tired', 'exhausted', 'burnout')

2. POSITIVE HABITS: Extract wellness activities, hobbies, or healthy behaviors (e.g., 'exercise', 'meditation', 'yoga', 'reading', 'walk', 'run', 'sport', 'family time',
    'friends', 'social', 'hobby', 'relax', 'rest', 'sleep', 'healthy', 'eat well')

3. EMOTIONAL TRENDS: List emotional words directly mentioned (e.g., 'happy', 'sad', 'angry', 'anxious', 'excited', 'nervous', 'calm', 'stressed',
    'depressed', 'joyful', 'frustrated', 'peaceful', 'worried', 'content')

4. SLEEP MENTION COUNT: Count all sleep-related phrases (sleep, slept, sleeping, rested, tired, etc.)

5. NUMBER OF CHILDREN: Extract numerical count of children mentioned (default to 0 if none)

6. MARITAL STATUS: Infer from family references (married, single, divorced, widowed) - use "unknown" if unclear

7. Sentiment: Analyze the sentiment of the text among POSITIVE, NEGATIVE and NEUTRAL

Be precise and only extract information explicitly stated or clearly implied. If any of these information you are able to find, return None or empty list [] instead of providing more general outupt.

Be very strict while extracing information from text, DO NOT ADD ANYTHING BY YOURSELF.

OUTPUT FORMAT: Return structured JSON with the exact field names as specified.
"""

def extract_wellness_profile(text: str) -> dict:
    
    if not text or not text.strip():
        return {
            "profile": {
                "stress_triggers": [],
                "positive_habits": [],
                "emotional_trends": [],
                "sleep_mention_count": 0,
                "number_of_children": 0,
                "marital_status": "unknown"
            },
            "confidence": 0.0
        }

    else:
        model = ChatOpenAI(model = "gpt-4o", temperature = 0).with_structured_output(ExtractionResult)
        messages = f"You are a precise NLP extraction system that analyzes wellness journal entries and returns structured JSON output. Be accurate and conservative in your extractions. \n\n  {create_extraction_prompt(text)}"    
        
        result = model.invoke(messages)
        return result


def main(text):

    test_result = extract_wellness_profile(text)
    print("Test Result:")
    print(test_result.model_dump_json())
    return test_result.model_dump_json()

text = """
Yesterday I felt anxious due to work deadlines. Today I watched TV with my wife and
later went playing cricket with my 2 kids. I slept poorly again.
"""
main(text)