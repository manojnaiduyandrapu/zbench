#!/usr/bin/env python3
"""
Extract structured resume data using Groq API to create organized sections.
For job descriptions: Extract overview, qualifications, technical requirements, and responsibilities
For resume content: Extract summary, technical skills, certifications, and projects
in specific structured sections.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_structured_job_description(job_description: str) -> str:
    """Extract job description into structured sections."""
    prompt = f"""
Extract and organize the following job description content into these specific sections:

**JOB OVERVIEW**:
- Position title and level
- Department/team
- Brief job purpose

**REQUIRED QUALIFICATIONS**:
- Education requirements
- Years of experience needed
- Required certifications
- Must-have skills and technologies

**TECHNICAL REQUIREMENTS**:
- Programming languages
- Frameworks and libraries
- Tools and technologies
- Databases
- Cloud platforms
- Other technical skills

**RESPONSIBILITIES**:
- Key duties and responsibilities
- Daily tasks and activities
- Leadership/collaboration requirements

Be concise and organized. Use bullet points for clarity. Only include information that is present in the job description.

Job Description:
{job_description}

Return the structured job description in the exact format above:
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
        print(f"Error extracting structured job description: {e}")
        return job_description[:1000]  # Fallback to truncation

def extract_structured_resume(resume_content: str) -> str:
    """Extract resume content into structured sections."""
    prompt = f"""
Extract and organize the following resume content into these specific sections:

**SUMMARY** (3-4 lines):
What the candidate is (role/title), Years of experience and Key expertise areas. (should be in such a way that it can be used as a summary in a resume)

**TECHNICAL SKILLS & PROGRAMMING**:
- Programming languages
- Frameworks and libraries
- Tools and technologies
- Databases
- Cloud platforms
- Other technical skills

**CERTIFICATIONS**:
- All professional certifications
- Training programs
- Industry credentials

**PROJECTS**:
- All projects the candidate has worked on that are relevant, include all the projects
- Include project descriptions
- Technologies used in each project
- Role and contributions

Be concise and organized. Use bullet points for clarity. Only include information that is present in the resume. Also ensure you have all sections present, even if some are empty.

Resume Content:
{resume_content}

Return the structured resume in the exact format above:
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
        print(f"Error extracting structured resume: {e}")
        return resume_content[:1000]  # Fallback to truncation

def process_jsonl_file(input_file: str, output_file: str):
    """Process the JSONL file and extract structured resume data."""
    print(f"Processing {input_file}...")
    
    processed_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract structured job description
                if 'query' in data and 'query' in data['query']:
                    print(f"Extracting structured job description for line {line_num}...")
                    data['query']['query'] = extract_structured_job_description(data['query']['query'])
                
                # Extract structured resume content in documents
                if 'documents' in data:
                    for i, doc in enumerate(data['documents']):
                        if 'content' in doc:
                            print(f"Extracting structured resume {i+1} for line {line_num}...")
                            doc['content'] = extract_structured_resume(doc['content'])
                
                # Write structured data
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write original line in case of error
                outfile.write(line)
    
    print(f"Processed {processed_count} entries. Output saved to {output_file}")

def main():
    """Main function to run the structured extraction."""
    input_file = "combine_resume_data.jsonl"
    output_file = "v3_llm_resume_data.jsonl"
    
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
    
    print(f"\nStructured extraction complete! Check {output_file} for results.")
    
    # Show file size comparison
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    processed_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    print(f"\nFile size comparison:")
    print(f"Original: {original_size:.2f} MB")
    print(f"Structured: {processed_size:.2f} MB")

if __name__ == "__main__":
    main()