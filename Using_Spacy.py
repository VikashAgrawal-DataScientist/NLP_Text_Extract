import re
import json
import spacy
from transformers import pipeline
from typing import List, Dict, Any


# spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

STRESS_TRIGGERS_KEYWORDS = {
    'work', 'deadline', 'pressure', 'stress', 'anxious', 'worried', 'overwhelmed',
    'busy', 'rush', 'urgent', 'behind', 'tired', 'exhausted', 'burnout'
}

POSITIVE_HABITS_KEYWORDS = {
    'exercise', 'meditation', 'yoga', 'reading', 'walk', 'run', 'sport', 'family time',
    'friends', 'social', 'hobby', 'relax', 'rest', 'sleep', 'healthy', 'eat well'
}

EMOTIONAL_KEYWORDS = {
    'happy', 'sad', 'angry', 'anxious', 'excited', 'nervous', 'calm', 'stressed',
    'depressed', 'joyful', 'frustrated', 'peaceful', 'worried', 'content'
}

def extract_stress_triggers(text: str) -> List[str]:
    doc = nlp(text.lower())
    triggers = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(trigger in sent_text for trigger in STRESS_TRIGGERS_KEYWORDS):
            for chunk in sent.noun_chunks:
                chunk_text = chunk.text.lower()
                if any(trigger in chunk_text for trigger in STRESS_TRIGGERS_KEYWORDS):
                    triggers.append(chunk_text)
    
    return list(set(triggers))


def extract_positive_habits(text: str) -> List[str]:
    doc = nlp(text.lower())
    habits = []
    
    activities = {
        'cricket': 'sports', 'tv': 'relaxation', 'meditation': 'meditation',
        'reading': 'reading', 'walk': 'walking', 'run': 'running', 'exercise': 'exercise',
        'yoga': 'yoga', 'family': 'family time', 'wife': 'family time', 'husband': 'family time',
        'kids': 'family time', 'children': 'family time'
    }
    
    for token in doc:
        if token.text in activities:
            habits.append(activities[token.text])
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(habit in sent_text for habit in POSITIVE_HABITS_KEYWORDS):
            habits.extend([habit for habit in POSITIVE_HABITS_KEYWORDS if habit in sent_text])
    
    return list(set(habits))

def extract_emotional_trends(text: str) -> List[str]:
    doc = nlp(text)
    emotions = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        found_emotions = [emotion for emotion in EMOTIONAL_KEYWORDS if emotion in sent_text]
        emotions.extend(found_emotions)
    
    try:
        sentiment_result = sentiment_analyzer(text[:512])
        if sentiment_result[0]['label'] == 'NEGATIVE' and sentiment_result[0]['score'] > 0.7:
            emotions.append('negative')
        elif sentiment_result[0]['label'] == 'POSITIVE' and sentiment_result[0]['score'] > 0.7:
            emotions.append('positive')
    except:
        pass
    
    return list(set(emotions))

def count_sleep_mentions(text: str) -> int:
    sleep_patterns = [
        r'slept?\s+\w+', r'sleep\s+\w+', r'slept', r'sleeping', r'sleepy',
        r'poorly slept', r'well slept', r'bad sleep', r'good sleep'
    ]
    
    count = 0
    text_lower = text.lower()
    
    for pattern in sleep_patterns:
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    
    return count

def extract_number_of_children(text: str) -> int:
    patterns = [
        r'my\s+(\d+)\s+kids?',
        r'(\d+)\s+children',
        r'(\d+)\s+kids?',
        r'kid?s?\s+(\d+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[0])
    
    child_indicators = ['kids', 'children', 'son', 'daughter']
    if any(indicator in text.lower() for indicator in child_indicators):
        return 1  
    
    return 0

def extract_marital_status(text: str) -> str:
    text_lower = text.lower()
    
    married_indicators = ['wife', 'husband', 'spouse', 'married']
    single_indicators = ['girlfriend', 'boyfriend', 'dating', 'single']
    
    if any(indicator in text_lower for indicator in married_indicators):
        return "married"
    elif any(indicator in text_lower for indicator in single_indicators):
        return "single"
    
    return "unknown"

def analyze_wellness_text(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return {
            "stress_triggers": [],
            "positive_habits": [],
            "emotional_trends": [],
            "sleep_mention_count": 0,
            "number_of_children": 0,
            "marital_status": "unknown"
        }
    
    profile_data = {
        "stress_triggers": extract_stress_triggers(text),
        "positive_habits": extract_positive_habits(text),
        "emotional_trends": extract_emotional_trends(text),
        "sleep_mention_count": count_sleep_mentions(text),
        "number_of_children": extract_number_of_children(text),
        "marital_status": extract_marital_status(text)
    }
    
    return profile_data

def process_multiple_entries(text_entries: List[str]) -> List[Dict[str, Any]]:
    return [analyze_wellness_text(entry) for entry in text_entries]

def main(text):
    
    print("Input Text")
    print(text)
    print("\n" + "="*50 + "\n")

    result = analyze_wellness_text(text)
    print("Extracted Profile Data")
    print(json.dumps(result, indent=2))
    return json.dumps(result, indent=2) 
   

text = """
Yesterday I felt anxious due to work deadlines. Today I watched TV with my wife and
later went playing cricket with my 2 kids. I slept poorly again.
"""

if __name__ == "__main__":
    main(text)