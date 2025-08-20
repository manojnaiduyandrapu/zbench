#!/usr/bin/env python3
"""
Compress resume data using Groq API to reduce content size by keeping only essential information.
For job descriptions: Keep requirements, responsibilities, and key qualifications
For resume content: Keep only skills, relevant experience, and important achievements
"""

import json
import os
from groq import Groq
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def compress_job_description(job_description: str) -> str:
    """Compress job description to keep only essential information."""
    prompt = f"""
Compress the following job description by keeping only the most essential information:
- Key responsibilities and requirements
- Required skills and technologies
- Experience requirements
- Important qualifications

Remove:
- Company descriptions and fluff
- Repetitive information
- Unnecessary details
- Long paragraphs about company culture

Job Description:
{job_description}

Return a concise version with only the essential information:
"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error compressing job description: {e}")
        return job_description[:1000]  # Fallback to truncation

def compress_resume_content(resume_content: str) -> str:
    """Compress resume content to keep only skills and relevant experience."""
    prompt = f"""
Extract and compress the following resume content to keep only:
- Technical skills and technologies
- Years of experience and key roles
- Important projects and achievements
- Relevant certifications
- Programming languages and tools

Remove:
- Contact information and personal details
- Long job descriptions
- Repetitive information
- Education details (unless highly relevant)
- Verbose explanations

Resume Content:
{resume_content}

Return a concise summary focused on skills and experience:
"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error compressing resume content: {e}")
        return resume_content[:800]  # Fallback to truncation

def process_jsonl_file(input_file: str, output_file: str):
    """Process the JSONL file and compress content."""
    print(f"Processing {input_file}...")
    
    processed_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Compress job description
                if 'query' in data and 'query' in data['query']:
                    print(f"Compressing job description for line {line_num}...")
                    data['query']['query'] = compress_job_description(data['query']['query'])
                
                # Compress resume content in documents
                if 'documents' in data:
                    for i, doc in enumerate(data['documents']):
                        if 'content' in doc:
                            print(f"Compressing resume {i+1} for line {line_num}...")
                            doc['content'] = compress_resume_content(doc['content'])
                
                # Write compressed data
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write original line in case of error
                outfile.write(line)
    
    print(f"Processed {processed_count} entries. Output saved to {output_file}")

def main():
    """Main function to run the compression."""
    input_file = "combine_resume_data.jsonl"
    output_file = "updated_llm_resume_data.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Check if GROQ_API_KEY is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set!")
        print("Please set your Groq API key: export GROQ_API_KEY=your_api_key_here")
        return
    
    # Process the file
    process_jsonl_file(input_file, output_file)
    
    print(f"\nCompression complete! Check {output_file} for results.")
    
    # Show file size comparison
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    compressed_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\nFile size comparison:")
    print(f"Original: {original_size:.2f} MB")
    print(f"Compressed: {compressed_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.1f}%")

if __name__ == "__main__":
    main()