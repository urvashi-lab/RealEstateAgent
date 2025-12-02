import os
import json
import re
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_intent_with_llm(user_query):
    prompt = f"""You must return ONLY a JSON object. Do NOT write code, explanations, or markdown.

User query: "{user_query}"

Extract these fields:
- areas: List of location names from the query
- metric: Must be EXACTLY one of: price, total_sales, flat_sold, office_sold, demand, units
- time_range: Time period mentioned (e.g., "last 5 years") or null

Metric selection rules:
- "price", "rate", "cost" → "price"
- "sales", "revenue" → "total_sales"
- "flat sold", "apartment sold" → "flat_sold"
- "office sold" , "offices → "office_sold"
- "shop sold", "shops sold", "retail sold", "store sold" ,"shops" → "shop_sold"
- "demand" → "demand"
- "units" → "units"

Return format (copy this structure):
{{"areas": ["area_name"], "metric": "metric_name", "time_range": null}}

Your JSON response:"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a JSON extraction bot. You ONLY return valid JSON objects. Never return code, explanations, or markdown. Just the JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0,  # ✅ Set to 0 for deterministic output
            max_tokens=150   # ✅ Limit tokens to prevent code generation
        )

        raw_output = completion.choices[0].message.content.strip()
        print(f"DEBUG - LLM Raw Output: {raw_output}")  # ✅ DEBUG
        
        # Remove any markdown formatting
        if "```" in raw_output:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_output, re.DOTALL)
            if match:
                raw_output = match.group(1).strip()
        
        # Remove "python" or other language indicators
        if raw_output.startswith(("python", "json", "javascript")):
            raw_output = re.sub(r'^[a-z]+\s*\n', '', raw_output, flags=re.IGNORECASE)
        
        # Try to extract JSON from text if LLM added extra text
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            raw_output = json_match.group(0)
        
        # Parse JSON
        intent = json.loads(raw_output)
        
        # Ensure areas is a list
        if "areas" not in intent:
            intent["areas"] = []
        elif isinstance(intent["areas"], str):
            intent["areas"] = [intent["areas"]]
        
        # Validate metric
        valid_metrics = ["price", "total_sales", "flat_sold", "office_sold", "demand", "units"]
        if "metric" not in intent or intent["metric"] not in valid_metrics:
            # Try to infer from query as fallback
            if any(word in user_query.lower() for word in ["price", "rate", "cost"]):
                intent["metric"] = "price"
            elif any(word in user_query.lower() for word in ["flat", "apartment"]):
                intent["metric"] = "flat_sold"
            else:
                intent["metric"] = "total_sales"  # Default
        
        return intent
        
    except json.JSONDecodeError as e:
        print(f"ERROR - Failed to parse JSON: {raw_output}")
        # Fallback: basic extraction
        areas = []
        metric = "price"  # default
        time_range = None
        
        # Extract areas (words that might be locations)
        words = user_query.lower().split()
        common_words = {"show", "me", "the", "in", "for", "of", "and", "last", "years", "year"}
        areas = [w for w in words if w not in common_words and len(w) > 2]
        
        # Extract metric
        if any(w in user_query.lower() for w in ["price", "rate", "cost"]):
            metric = "price"
        elif "flat" in user_query.lower():
            metric = "flat_sold"
        elif "office" in user_query.lower():
            metric = "office_sold"
        elif "sales" in user_query.lower():
            metric = "total_sales"
        
        # Extract time range
        time_match = re.search(r'last (\d+) years?', user_query.lower())
        if time_match:
            time_range = f"last {time_match.group(1)} years"
        
        return {
            "areas": areas[:3],  # Take first 3 potential areas
            "metric": metric,
            "time_range": time_range
        }
    except Exception as e:
        raise ValueError(f"Failed to extract intent: {str(e)}")
    
