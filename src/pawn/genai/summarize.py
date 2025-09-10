#!/usr/bin/env python3

import os
import re

def sanitize_input(text):
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s\-\.]', '', text)[:200]

def generate_summary(dept, week, drivers):
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        try:
            import openai
            openai.api_key = api_key
            
            safe_dept = sanitize_input(dept)
            safe_drivers = [sanitize_input(d) for d in drivers[:3]]
            
            prompt = f"Department {safe_dept} risk factors: {', '.join(safe_drivers)}. Generate 3 brief bullets."
            
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].text.strip().split('\n')[:3]
        except:
            pass
    
    return rule_based_summary(dept, drivers)

def rule_based_summary(dept, drivers):
    bullets = []
    
    if 'overtime' in str(drivers).lower():
        bullets.append("• Excessive overtime detected - review workload distribution")
    
    if 'satisfaction' in str(drivers).lower():
        bullets.append("• Low satisfaction scores - consider team engagement initiatives")
    
    if 'stagnation' in str(drivers).lower():
        bullets.append("• Career stagnation risk - evaluate promotion opportunities")
    
    if not bullets:
        bullets = [
            "• Monitor department metrics closely",
            "• Review recent policy changes impact", 
            "• Consider preventive HR interventions"
        ]
    
    return bullets[:3]
