import asyncio
from zbench.annotation import EnsembleZELOAnnotator

async def main():
    # Set up the annotator
    annotator = EnsembleZELOAnnotator(
        dataset_path="/Users/manojnaiduyandrapu/Documents/zbench/job_resume_data.jsonl",           # Your input file
        annotated_dataset_path="/Users/manojnaiduyandrapu/Documents/zbench/job_resumes_ranked.jsonl",  # Output file
        cycle_num=4,                                # Number of comparison rounds
        document_limit=None                         # Use all 100 resumes
    )
    
    print("🚀 Starting resume ranking...")
    await annotator.zelo_annotate()
    print("✅ Ranking complete!")

if __name__ == "__main__":
    asyncio.run(main())