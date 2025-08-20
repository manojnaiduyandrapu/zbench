#!/usr/bin/env python3
"""
Create a new JSONL file with compressed resume data using Groq API.
Replaces verbose job descriptions and resume content with essential information only.
"""

import json
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def compress_job_description(job_description: str) -> str:
    """Extract essential job requirements and responsibilities."""
    prompt = f"""Extract only the essential information from this job description:

{job_description}

Return ONLY:
- Required skills and technologies (bullet points)
- Key responsibilities (bullet points)  
- Experience requirements
- Must-have qualifications

Keep it under 200 words, focus on what matters for matching candidates."""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error compressing job description: {e}")
        return job_description[:500]

def compress_resume_content(resume_content: str) -> str:
    """Extract essential skills and experience from resume."""
    prompt = f"""Extract only the essential information from this resume:

{resume_content}

Return ONLY:
- Technical skills and tools (comma-separated)
- Years of experience in key areas
- Important technologies worked with
- Relevant certifications
- Key achievements (1-2 lines max)

Keep it under 150 words, focus on matchable skills and experience."""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=250,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error compressing resume content: {e}")
        return resume_content[:400]

def main():
    input_file = "combine_resume_data.jsonl"
    output_file = "compressed_resume_data.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
        
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: Please set GROQ_API_KEY environment variable")
        return
    
    print(f"Processing {input_file}...")
    processed = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Compress job description
                if 'query' in data and 'query' in data['query']:
                    print(f"Line {line_num}: Compressing job description...")
                    data['query']['query'] = compress_job_description(data['query']['query'])
                
                # Compress each resume
                if 'documents' in data:
                    for i, doc in enumerate(data['documents']):
                        if 'content' in doc:
                            print(f"Line {line_num}: Compressing resume {i+1}...")
                            doc['content'] = compress_resume_content(doc['content'])
                
                outfile.write(json.dumps(data) + '\n')
                processed += 1
                
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                outfile.write(line)
    
    print(f"\nCreated {output_file} with {processed} compressed entries")
    
    # Show size comparison
    original_size = os.path.getsize(input_file) / 1024
    new_size = os.path.getsize(output_file) / 1024
    print(f"Original: {original_size:.1f} KB")
    print(f"Compressed: {new_size:.1f} KB")
    print(f"Reduction: {((original_size - new_size) / original_size * 100):.1f}%")

if __name__ == "__main__":
    main()