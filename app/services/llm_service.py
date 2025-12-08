import os
import json
from groq import Groq

def get_disease_recommendations(crop: str, disease: str, language: str = "en") -> list:
    """
    Generate disease-specific recommendations using Groq LLM.
    
    Args:
        crop: Detected crop type (e.g., "Apple", "Tomato")
        disease: Detected disease (e.g., "Black_rot", "Early_blight")
        language: Language code (en, hi, ta, te)
    
    Returns:
        List of actionable recommendations
    """
    try:
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY not set, using fallback recommendations")
            return get_fallback_recommendations(crop, disease, language)
        
        client = Groq(api_key=api_key)
        
        # Language mapping for prompts
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu"
        }
        
        lang_name = language_names.get(language, "English")
        
        # Create prompt for LLM
        prompt = f"""You are an expert agricultural advisor. A farmer has detected {disease} on their {crop} plant.

Provide 5-6 specific, actionable recommendations for treating this disease in {lang_name}.

Requirements:
- Each recommendation should be 1-2 sentences
- Focus on immediate actions, treatment methods, and prevention
- Be specific to {crop} and {disease}
- Use clear, farmer-friendly language
- Return ONLY a JSON array of strings (no other text)

Example format: ["Recommendation 1", "Recommendation 2", ...]
"""
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agricultural advisor providing disease treatment advice to farmers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-70b-versatile",  # Fast and capable model
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract recommendations from response
        recommendations_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            recommendations = json.loads(recommendations_text)
            if isinstance(recommendations, list) and len(recommendations) > 0:
                return recommendations[:6]  # Limit to 6 recommendations
        except json.JSONDecodeError:
            # If not valid JSON, split by newlines and clean up
            lines = recommendations_text.split('\n')
            recommendations = [
                line.strip().lstrip('-').lstrip('•').lstrip('*').lstrip('1234567890.').strip()
                for line in lines 
                if line.strip() and not line.strip().startswith('[') and not line.strip().startswith(']')
            ]
            recommendations = [r for r in recommendations if len(r) > 10][:6]
            
        return recommendations if recommendations else get_fallback_recommendations(crop, disease, language)
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return get_fallback_recommendations(crop, disease, language)


def get_fallback_recommendations(crop: str, disease: str, language: str) -> list:
    """
    Fallback recommendations if LLM fails or API key not set.
    """
    fallback = {
        "en": [
            f"Remove and destroy infected {crop.lower()} plant parts immediately",
            f"Apply appropriate fungicide or pesticide for {disease.replace('_', ' ')}",
            "Improve air circulation between plants",
            "Avoid overhead watering to reduce moisture on leaves",
            "Monitor neighboring plants for similar symptoms",
            "Consult local agricultural extension officer for treatment"
        ],
        "hi": [
            f"संक्रमित {crop} पौधे के हिस्सों को तुरंत हटा दें और नष्ट करें",
            f"{disease.replace('_', ' ')} के लिए उपयुक्त फफूंदनाशक या कीटनाशक लगाएं",
            "पौधों के बीच वायु संचलन में सुधार करें",
            "पत्तियों पर नमी कम करने के लिए ऊपरी सिंचाई से बचें",
            "समान लक्षणों के लिए पड़ोसी पौधों की निगरानी करें",
            "उपचार के लिए स्थानीय कृषि विस्तार अधिकारी से परामर्श करें"
        ]
    }
    
    return fallback.get(language, fallback["en"])
