from __future__ import annotations
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

from .upload_dataset import create_dataset, upload_dataset_file
from .start_batch_job import create_batch_job
from .poll_and_download import poll_until_done, get_dataset, try_download_external_url
from config.schema import load_config


@dataclass
class JobInfo:
    """Information about a batch job"""
    part_number: int
    dataset_id: str
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    status: str = "pending"  # pending, submitted, running, completed, failed
    submit_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    results_path: Optional[str] = None


@dataclass
class QueueManager:
    """Manages batch job queue with concurrency limits"""
    account_id: str
    model_id: str
    config: dict
    max_concurrent: int = 4
    poll_interval: int = 30  # seconds
    jobs: List[JobInfo] = field(default_factory=list)
    running_jobs: Dict[str, JobInfo] = field(default_factory=dict)
    temp_label: str = "10"
    temperature: float = 1.0
    condition: str = "treatment"
    run_id: str = ""
    
    def add_job(self, part_number: int, dataset_path: str, dataset_id: str = None) -> JobInfo:
        """Add a job to the queue"""
        job = JobInfo(
            part_number=part_number,
            dataset_id=dataset_id or f"temp-dataset-p{part_number:02d}"
        )
        self.jobs.append(job)
        return job
    
    def can_submit_more(self) -> bool:
        """Check if we can submit more jobs"""
        return len(self.running_jobs) < self.max_concurrent
    
    def get_next_pending(self) -> Optional[JobInfo]:
        """Get the next pending job"""
        for job in self.jobs:
            if job.status == "pending":
                return job
        return None
    
    def submit_job(self, job: JobInfo) -> bool:
        """Submit a single job"""
        try:
            print(f"Submitting job for part P{job.part_number:02d}...")
            
            # Create batch job
            job_response = create_batch_job(
                account_id=self.account_id,
                model=self.config["model_id"],
                input_dataset_id=job.dataset_id,
                display_name=f"excellence-t{self.temp_label}-{self.condition}-{self.run_id}-p{job.part_number:02d}",
                temperature=float(self.temperature),
                max_tokens=self.config.get("max_new_tokens", {}).get("open_book", 1024),
                top_p=self.config.get("top_p"),
                top_k=self.config.get("top_k"),
                stop=self.config.get("stop")
            )
            
            job.job_id = job_response.get("id")
            job.job_name = job_response.get("name")
            job.status = "submitted"
            job.submit_time = datetime.now()
            
            # Add to running jobs
            if job.job_name:
                self.running_jobs[job.job_name] = job
                print(f"✓ Job P{job.part_number:02d} submitted: {job.job_name}")
                return True
            else:
                print(f"✗ Failed to get job name for P{job.part_number:02d}")
                job.status = "failed"
                return False
                
        except Exception as e:
            print(f"✗ Failed to submit job P{job.part_number:02d}: {e}")
            job.status = "failed"
            return False
    
    def check_running_jobs(self) -> List[JobInfo]:
        """Check status of running jobs and return completed ones"""
        completed = []
        
        for job_name, job in list(self.running_jobs.items()):
            try:
                job_result = poll_until_done(self.account_id, job_name, poll_seconds=0)  # Just check once
                
                job_state = job_result.get("state")
                if job_state in ("COMPLETED", "FAILED", "EXPIRED"):
                    # Job finished
                    del self.running_jobs[job_name]
                    job.complete_time = datetime.now()
                    
                    if job_state == "COMPLETED":
                        job.status = "completed"
                        print(f"✓ Job P{job.part_number:02d} completed")
                    else:
                        job.status = "failed"
                        print(f"✗ Job P{job.part_number:02d} failed: {job_state}")
                    
                    completed.append(job)
                else:
                    job.status = "running"
                    
            except Exception as e:
                print(f"Warning: Error checking job P{job.part_number:02d}: {e}")
        
        return completed
    
    def download_results(self, job: JobInfo, results_dir: str) -> bool:
        """Download results for a completed job"""
        if job.status != "completed":
            return False
            
        try:
            print(f"Downloading results for P{job.part_number:02d}...")
            
            # Get job details again to get output dataset
            job_result = poll_until_done(self.account_id, job.job_name)
            out_ds_id = job_result.get("outputDatasetId") or job_result.get("output_dataset_id")
            
            if not out_ds_id:
                print(f"No output dataset for P{job.part_number:02d}")
                return False
            
            # Create job-specific directory
            job_dir = os.path.join(results_dir, f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}")
            os.makedirs(job_dir, exist_ok=True)
            
            # Get dataset and download
            ds = get_dataset(self.account_id, out_ds_id.split("/")[-1])
            ext_url = ds.get("externalUrl") or ds.get("external_url")
            
            if ext_url:
                downloaded_path = try_download_external_url(ext_url, job_dir)
                if downloaded_path:
                    # Try to extract JSONL files
                    from .poll_and_download import _try_extract_jsonls, _combine_jsonls
                    extracted = _try_extract_jsonls(downloaded_path, job_dir)
                    
                    if extracted:
                        # Combine extracted files
                        combined_path = os.path.join(job_dir, "results.jsonl")
                        n_lines = _combine_jsonls(job_dir, combined_path)
                        
                        if n_lines > 0:
                            job.results_path = combined_path
                            print(f"✓ Downloaded {n_lines} results for P{job.part_number:02d}")
                            return True
            
            print(f"✗ Failed to download results for P{job.part_number:02d}")
            return False
            
        except Exception as e:
            print(f"✗ Error downloading P{job.part_number:02d}: {e}")
            return False
    
    def print_status(self):
        """Print current queue status"""
        pending = sum(1 for j in self.jobs if j.status == "pending")
        running = len(self.running_jobs)
        completed = sum(1 for j in self.jobs if j.status == "completed")
        failed = sum(1 for j in self.jobs if j.status == "failed")
        
        print(f"\n=== Queue Status ===")
        print(f"Pending: {pending}, Running: {running}, Completed: {completed}, Failed: {failed}")
        
        if self.running_jobs:
            print("Currently running:")
            for job_name, job in self.running_jobs.items():
                if job.submit_time:
                    elapsed = (datetime.now() - job.submit_time).total_seconds() / 60
                    print(f"  P{job.part_number:02d}: {elapsed:.1f} min")
                else:
                    print(f"  P{job.part_number:02d}: submitting…")
    
    def run_queue(self, results_dir: str):
        """Main loop to process the job queue"""
        print(f"Starting queue processing with max {self.max_concurrent} concurrent jobs")
        
        while True:
            # Submit new jobs if possible
            while self.can_submit_more():
                next_job = self.get_next_pending()
                if next_job is None:
                    break  # No more pending jobs
                
                self.submit_job(next_job)
                time.sleep(2)  # Brief pause between submissions
            
            # Check running jobs
            completed_jobs = self.check_running_jobs()
            
            # Download results for completed jobs
            for job in completed_jobs:
                if job.status == "completed":
                    self.download_results(job, results_dir)
            
            # Print status
            self.print_status()
            
            # Check if we're done
            all_done = all(j.status in ("completed", "failed") for j in self.jobs)
            if all_done:
                print("\n✓ All jobs completed!")
                break
            
            # Wait before next check
            if self.running_jobs:
                print(f"Waiting {self.poll_interval}s before next check...")
                time.sleep(self.poll_interval)
            else:
                print("No running jobs, checking again in 5s...")
                time.sleep(5)


def upload_datasets(account_id: str, dataset_files: List[Tuple[int, str]], base_name: str, temp_label: str, condition: str) -> List[Tuple[int, str]]:
    """Upload datasets and return list of (part_number, dataset_id).

    dataset_files: list of (part_number, local_path)
    """
    results: List[Tuple[int, str]] = []
    for part_number, dataset_path in dataset_files:
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found: {dataset_path}")
            continue
        print(f"Uploading P{part_number:02d}: {os.path.basename(dataset_path)}")
        try:
            ds_name = f"{base_name}-p{part_number:02d}"
            dataset_id = create_dataset(ds_name, account_id)
            remote_fname = f"t{temp_label}_{condition}.p{part_number:02d}.jsonl"
            upload_dataset_file(account_id, dataset_id, dataset_path, filename=remote_fname)
            results.append((part_number, dataset_id))
            print(f"✓ Uploaded P{part_number:02d}")
        except Exception as e:
            print(f"✗ Failed to upload P{part_number:02d}: {e}")
    return results


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Upload dataset parts and process them with queue management")
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--account_id", default=os.environ.get("FIREWORKS_ACCOUNT_ID"))
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--start_part", type=int, default=1, help="Starting part number (default: 1)")
    parser.add_argument("--end_part", type=int, default=15, help="Ending part number (default: 15)")
    parser.add_argument("--max_concurrent", type=int, default=4, help="Max concurrent jobs (default: 4)")
    parser.add_argument("--data_dir", default="data/batch_inputs")
    parser.add_argument("--condition", choices=["control", "treatment"], default="treatment")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--run_id", default=None, help="Run identifier; default=timestamp")
    
    args = parser.parse_args()
    
    if not args.account_id:
        print("Error: FIREWORKS_ACCOUNT_ID must be set")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Build naming components
    t_str = ("0" if f"{float(args.temp):.1f}" == "0.0" else f"{float(args.temp):.1f}".replace(".", ""))
    run_id = args.run_id or datetime.utcnow().strftime("r%Y%m%d%H%M%S")
    base_name = f"excellence-t{t_str}-{args.condition}-{run_id}"
    
    # Build list of dataset files to process (expect split parts named t{t_str}_{condition}.pXX.jsonl)
    dataset_files: List[Tuple[int, str]] = []
    for i in range(args.start_part, args.end_part + 1):
        path = os.path.join(args.data_dir, f"t{t_str}_{args.condition}.p{i:02d}.jsonl")
        dataset_files.append((i, path))
    
    print(f"Processing parts P{args.start_part:02d} through P{args.end_part:02d}")
    print(f"Dataset files: {len(dataset_files)} files")
    
    # Upload datasets
    uploaded_datasets = upload_datasets(args.account_id, dataset_files, base_name, t_str, args.condition)
    
    if not uploaded_datasets:
        print("No datasets uploaded successfully!")
        sys.exit(1)
    
    print(f"\nSuccessfully uploaded {len(uploaded_datasets)} datasets")
    
    # Create queue manager
    queue_mgr = QueueManager(
        account_id=args.account_id,
        model_id=config["model_id"],
        config=config,
        max_concurrent=args.max_concurrent,
        temp_label=t_str,
        temperature=float(args.temp),
        condition=args.condition,
        run_id=run_id
    )
    
    # Add jobs to queue
    for part_num, dataset_id in uploaded_datasets:
        queue_mgr.add_job(part_num, "", dataset_id)
    
    print(f"\nStarting batch processing with {len(queue_mgr.jobs)} jobs...")
    
    # Process the queue
    try:
        queue_mgr.run_queue(args.results_dir)
        
        # Print final summary
        completed = sum(1 for j in queue_mgr.jobs if j.status == "completed")
        failed = sum(1 for j in queue_mgr.jobs if j.status == "failed")
        
        print(f"\n=== Final Results ===")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed jobs:")
            for job in queue_mgr.jobs:
                if job.status == "failed":
                    print(f"  P{job.part_number:02d}")
        
        print("\nAll datasets processed!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        queue_mgr.print_status()


if __name__ == "__main__":
    main()