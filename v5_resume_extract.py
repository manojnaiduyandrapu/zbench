#!/usr/bin/env python3
"""
Extract structured resume data using Groq API to create organized sections.
For job descriptions: Use whole content as query (extraction commented out)
For resume content: Extract summary, technical skills, certifications, and projects
in specific structured sections.
"""

import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class JobOverview(BaseModel):
    overview_text: str = Field(description="Job overview in sentence format including position title, department/team, and brief job purpose")

class RequiredQualifications(BaseModel):
    education_requirements: List[str] = Field(default=[], description="Education requirements")
    years_of_experience: Optional[str] = Field(default=None, description="Years of experience needed")
    required_certifications: List[str] = Field(default=[], description="Required certifications")
    must_have_skills: List[str] = Field(default=[], description="Must-have skills and technologies")

class TechnicalRequirements(BaseModel):
    programming_languages: List[str] = Field(default=[], description="Programming languages")
    frameworks_libraries: List[str] = Field(default=[], description="Frameworks and libraries")
    tools_technologies: List[str] = Field(default=[], description="Tools and technologies")
    databases: List[str] = Field(default=[], description="Databases")
    cloud_platforms: List[str] = Field(default=[], description="Cloud platforms")
    other_technical_skills: List[str] = Field(default=[], description="Other technical skills")

class Responsibilities(BaseModel):
    key_duties: List[str] = Field(default=[], description="Key duties and responsibilities")
    daily_tasks: List[str] = Field(default=[], description="Daily tasks and activities")
    leadership_collaboration: List[str] = Field(default=[], description="Leadership/collaboration requirements")

class StructuredJobDescription(BaseModel):
    job_overview: JobOverview
    required_qualifications: RequiredQualifications
    technical_requirements: TechnicalRequirements
    responsibilities: Responsibilities

class ResumeSummary(BaseModel):
    summary_text: str = Field(description="4-5 line summary including role/title, years of experience, and key expertise areas")

class TechnicalSkills(BaseModel):
    programming_languages: List[str] = Field(default=[], description="Programming languages")
    frameworks_libraries: List[str] = Field(default=[], description="Frameworks and libraries")
    tools_technologies: List[str] = Field(default=[], description="Tools and technologies")
    databases: List[str] = Field(default=[], description="Databases")
    cloud_platforms: List[str] = Field(default=[], description="Cloud platforms")
    other_technical_skills: List[str] = Field(default=[], description="Other technical skills")

class Certification(BaseModel):
    name: str = Field(description="Certification name")
    issuer: Optional[str] = Field(default=None, description="Issuing organization")
    date: Optional[str] = Field(default=None, description="Date obtained")

class Project(BaseModel):
    name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    technologies_used: List[str] = Field(default=[], description="Technologies used in project")
    role_contributions: str = Field(description="Role and contributions")

class StructuredResume(BaseModel):
    summary: ResumeSummary
    technical_skills: TechnicalSkills
    certifications: List[Certification] = Field(default=[], description="Professional certifications")
    projects: List[Project] = Field(default=[], description="Relevant projects")

def extract_structured_job_description(job_description: str) -> str:
    """Extract job description into structured sections using schema."""
    schema = StructuredJobDescription.model_json_schema()
    
    prompt = f"""
Extract and organize the following job description content according to this JSON schema:

{json.dumps(schema, indent=2)}

Job Description:
{job_description}

Return a valid JSON object that matches the schema above. Extract only information that is present in the job description. Use empty arrays for missing list fields.
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
        
        # Try to parse and validate the JSON response
        json_response = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        if not json_response:
            print("Error: Empty response from API")
            return job_description[:1000]  # Fallback to truncation
        
        # Extract JSON from markdown code blocks if present
        if "```json" in json_response:
            start = json_response.find("```json") + 7
            end = json_response.find("```", start)
            if end != -1:
                json_response = json_response[start:end].strip()
        elif "```" in json_response:
            start = json_response.find("```") + 3
            end = json_response.find("```", start)
            if end != -1:
                json_response = json_response[start:end].strip()
        
        try:
            parsed_data = json.loads(json_response)
            validated_data = StructuredJobDescription(**parsed_data)
            return json.dumps(validated_data.model_dump(), indent=2)
        except (json.JSONDecodeError, Exception) as parse_error:
            print(f"Error parsing JSON response: {parse_error}")
            return json_response
            
    except Exception as e:
        print(f"Error extracting structured job description: {e}")
        return job_description[:1000]  # Fallback to truncation

def extract_structured_resume(resume_content: str) -> str:
    """Extract resume content into structured sections using schema."""
    schema = StructuredResume.model_json_schema()
    
    prompt = f"""
Extract and organize the following resume content according to this JSON schema:

{json.dumps(schema, indent=2)}

Resume Content:
{resume_content}

Return a valid JSON object that matches the schema above. Extract only information that is present in the resume. Use empty arrays for missing list fields.

For the summary section, create a 3-4 line summary that includes:
- What the candidate is (role/title)
- Years of experience 
- Key expertise areas

For projects, include all relevant projects with descriptions, technologies used, and the candidate's role/contributions.
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
        
        # Try to parse and validate the JSON response
        json_response = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        if not json_response:
            print("Error: Empty response from API")
            return resume_content[:1000]  # Fallback to truncation
        
        # Extract JSON from markdown code blocks if present
        if "```json" in json_response:
            start = json_response.find("```json") + 7
            end = json_response.find("```", start)
            if end != -1:
                json_response = json_response[start:end].strip()
        elif "```" in json_response:
            start = json_response.find("```") + 3
            end = json_response.find("```", start)
            if end != -1:
                json_response = json_response[start:end].strip()
            
        try:
            parsed_data = json.loads(json_response)
            validated_data = StructuredResume(**parsed_data)
            return json.dumps(validated_data.model_dump(), indent=2)
        except (json.JSONDecodeError, Exception) as parse_error:
            print(f"Error parsing JSON response: {parse_error}")
            return json_response
            
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
                
                # Extract structured job description (COMMENTED OUT - use whole content as query)
                # if 'query' in data and 'query' in data['query']:
                #     print(f"Extracting structured job description for line {line_num}...")
                #     data['query']['query'] = extract_structured_job_description(data['query']['query'])
                
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
    output_file = "v5_llm_resume_data.jsonl"
    
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