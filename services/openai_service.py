import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter API configuration for GPT-OSS-20B
API_KEY = os.getenv("OPENROUTER_API_KEY")  # You'll need to add this to your .env file
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free"

if not API_KEY:
    print("Warning: OPENROUTER_API_KEY not found in environment variables")

SYSTEM_PROMPT = """You are an AI meeting assistant. Your role is to analyze raw transcripts and generate structured, concise notes.

Always format output in this structure:

**Executive Summary:**
Brief overview of the meeting purpose and main outcomes.

**Key Discussion Points:**
• Main topics discussed
• Important details and context

**Action Items:**
• Specific tasks with responsible parties (if mentioned)
• Deadlines (if mentioned)

**Decisions Made:**
• Clear decisions reached during the meeting
• Next steps agreed upon

Please be concise but comprehensive in your analysis."""

def summarize_with_openai(transcript: str) -> str:
    """
    Use OpenRouter API with GPT-OSS-20B to summarize transcript into structured notes.
    """
    if not transcript.strip():
        return "Error: No transcript provided to summarize."
    
    if not API_KEY:
        return "Error: OpenRouter API key not configured."
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",  # Replace with your domain
        "X-Title": "Meeting Notes AI"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please analyze this meeting transcript and create structured notes:\n\n{transcript}"}
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.8,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
    
    try:
        print(f"🚀 Sending request to OpenRouter API with model: {MODEL_NAME}")
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have a valid response
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
                
                if content:
                    print(f"✅ OpenAI GPT-OSS response received: {len(content)} characters")
                    return content.strip()
                else:
                    return "Error: Empty response from OpenAI GPT-OSS"
            else:
                print(f"❌ Invalid response format: {data}")
                return "Error: Invalid response format from OpenAI GPT-OSS"
                
        elif response.status_code == 401:
            return "Error: Invalid API key for OpenRouter"
        elif response.status_code == 402:
            return "Error: Insufficient credits in OpenRouter account"
        elif response.status_code == 429:
            return "Error: Rate limit exceeded for OpenRouter API"
        else:
            error_text = response.text
            print(f"❌ API Error {response.status_code}: {error_text}")
            return f"Error {response.status_code}: Failed to process with OpenAI GPT-OSS"
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return "Error: Request timed out while processing with OpenAI GPT-OSS"
    except requests.exceptions.RequestException as e:
        print(f"❌ Request exception: {str(e)}")
        return f"Error: Request failed - {str(e)}"
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return f"Error: Unexpected error - {str(e)}"

def get_openai_model_status():
    """Check if OpenAI GPT-OSS service is available"""
    return {
        'available': bool(API_KEY),
        'model': MODEL_NAME,
        'provider': 'openrouter'
    }

# Testing function
def test_openai_api():
    """Test the OpenAI GPT-OSS API connection"""
    test_transcript = """
    John: Hello team, let's discuss the Q3 budget. We need to allocate funds for the marketing campaign.
    Sarah: I think we should focus on digital marketing. Our ROI has been better there.
    Mike: Agreed. Let's allocate 60% to digital and 40% to traditional marketing.
    John: Perfect. Sarah, can you prepare a detailed plan by Friday?
    Sarah: Absolutely, I'll have it ready.
    """
    
    print("🧪 Testing OpenAI GPT-OSS API...")
    result = summarize_with_openai(test_transcript)
    print("Test Result:")
    print("=" * 50)
    print(result)
    print("=" * 50)

if __name__ == "__main__":
    test_openai_api()