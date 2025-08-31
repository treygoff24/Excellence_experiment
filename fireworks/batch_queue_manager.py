from __future__ import annotations
import os
import sys
import json
import time
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from typing import Callable, Any
from dotenv import load_dotenv

from .upload_dataset import create_dataset, upload_dataset_file
from .start_batch_job import create_batch_job
from .poll_and_download import poll_until_done, get_dataset, try_download_external_url, get_batch_job, _normalize_state
from config.schema import load_config


@dataclass
class JobInfo:
    """Information about a batch job with detailed state tracking"""
    part_number: int
    dataset_id: str
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    status: str = "pending"  # pending, submitted, running, completed, failed
    submit_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    results_path: Optional[str] = None
    last_known_state: Optional[str] = None
    state_transitions: List[Tuple[datetime, str]] = field(default_factory=list)
    retry_count: int = 0
    last_error: Optional[str] = None
    
    def add_state_transition(self, new_state: str):
        """Add a state transition with timestamp"""
        self.state_transitions.append((datetime.now(), new_state))
        self.last_known_state = new_state
        
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds if completed"""
        if self.submit_time and self.complete_time:
            return (self.complete_time - self.submit_time).total_seconds()
        return None


@dataclass
class QueueManager:
    """Manages batch job queue with concurrency limits and thread safety"""
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
    progress_cb: Optional[Callable[[dict], None]] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _submission_semaphore: Optional[threading.Semaphore] = None
    
    def __post_init__(self):
        """Initialize semaphore after dataclass creation"""
        self._submission_semaphore = threading.Semaphore(self.max_concurrent)
    
    def add_job(self, part_number: int, dataset_path: str, dataset_id: str = None) -> JobInfo:
        """Add a job to the queue (thread-safe)"""
        with self._lock:
            job = JobInfo(
                part_number=part_number,
                dataset_id=dataset_id or f"temp-dataset-p{part_number:02d}"
            )
            self.jobs.append(job)
            return job
    
    def can_submit_more(self) -> bool:
        """Check if we can submit more jobs (thread-safe)"""
        with self._lock:
            return len(self.running_jobs) < self.max_concurrent
    
    def get_next_pending(self) -> Optional[JobInfo]:
        """Get the next pending job (thread-safe)"""
        with self._lock:
            for job in self.jobs:
                if job.status == "pending":
                    return job
            return None
    
    def submit_job(self, job: JobInfo) -> bool:
        """Submit a single job (thread-safe with semaphore)"""
        # Acquire semaphore to limit concurrent submissions
        if not self._submission_semaphore.acquire(timeout=30.0):
            print(f"✗ Timeout acquiring semaphore for P{job.part_number:02d}")
            job.status = "failed"
            return False
            
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
            
            # Thread-safe update of job state and running_jobs
            with self._lock:
                job.job_id = job_response.get("id")
                job.job_name = job_response.get("name")
                job.status = "submitted"
                job.submit_time = datetime.now()
                job.add_state_transition("SUBMITTED")
                
                # Add to running jobs
                if job.job_name:
                    self.running_jobs[job.job_name] = job
                    print(f"✓ Job P{job.part_number:02d} submitted: {job.job_name}")
                    # Progress event with safe callback
                    self._safe_progress_callback({
                        "event": "submitted",
                        "job_key": f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}",
                        "job_name": job.job_name,
                        "dataset_id": job.dataset_id,
                        "timestamp": job.submit_time.isoformat(),
                    })
                    return True
                else:
                    print(f"✗ Failed to get job name for P{job.part_number:02d}")
                    job.status = "failed"
                    job.add_state_transition("FAILED")
                    job.last_error = "Failed to get job name from response"
                    self._submission_semaphore.release()  # Release on failure
                    return False
                    
        except Exception as e:
            print(f"✗ Failed to submit job P{job.part_number:02d}: {e}")
            with self._lock:
                job.status = "failed"
                job.add_state_transition("FAILED")
                job.last_error = str(e)
            self._submission_semaphore.release()  # Release on exception
            return False
    
    def _retry_with_backoff(self, operation, max_retries: int = 3, base_delay: float = 1.0):
        """Retry an operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                print(f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s due to: {e}")
                time.sleep(delay)

    def check_running_jobs(self) -> List[JobInfo]:
        """Check status of running jobs and return completed ones with retry logic (thread-safe)."""
        completed: List[JobInfo] = []
        
        # Get a snapshot of running jobs to avoid concurrent modification
        with self._lock:
            jobs_to_check = list(self.running_jobs.items())
        
        for job_name, job in jobs_to_check:
            try:
                # Use retry logic for job status check
                def _get_job_status():
                    return get_batch_job(self.account_id, job_name)
                
                job_result = self._retry_with_backoff(_get_job_status)
                job_state = _normalize_state(job_result.get("state"))
                
                if job_state in ("COMPLETED", "FAILED", "EXPIRED"):
                    # Job finished - thread-safe removal and semaphore release
                    with self._lock:
                        if job_name in self.running_jobs:  # Double-check still exists
                            del self.running_jobs[job_name]
                            job.complete_time = datetime.now()
                            job.add_state_transition(job_state)
                            
                            if job_state == "COMPLETED":
                                job.status = "completed"
                                duration = job.get_duration()
                                duration_str = f" ({duration:.1f}s)" if duration else ""
                                print(f"✓ Job P{job.part_number:02d} completed{duration_str}")
                            else:
                                job.status = "failed"
                                job.last_error = f"Job ended with state: {job_state}"
                                print(f"✗ Job P{job.part_number:02d} failed: {job_state}")
                            
                            # Release semaphore slot for new job
                            self._submission_semaphore.release()
                            completed.append(job)
                    
                    # State change event (terminal) - with error handling
                    self._safe_progress_callback({
                        "event": "state",
                        "job_key": f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}",
                        "job_name": job_name,
                        "state": job_state,
                        "timestamp": datetime.now().isoformat(),
                        "duration": job.get_duration(),
                        "transitions": len(job.state_transitions),
                    })
                else:
                    # Update status for running job
                    with self._lock:
                        if job_name in self.running_jobs:  # Still running
                            job.status = "running"
                            # Only add state transition if state changed
                            if job_state and job_state != job.last_known_state:
                                job.add_state_transition(job_state)
                    
                    # Running state event - with error handling (only if state changed)
                    if job_state and job_state != job.last_known_state:
                        self._safe_progress_callback({
                            "event": "state",
                            "job_key": f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}",
                            "job_name": job_name,
                            "state": job_state,
                            "timestamp": datetime.now().isoformat(),
                            "transitions": len(job.state_transitions),
                        })
            except Exception as e:
                print(f"Error checking job P{job.part_number:02d} after retries: {e}")
                # Don't remove from running_jobs - will retry on next check
        return completed
    
    def _safe_progress_callback(self, event_data: dict):
        """Safely call progress callback with error logging."""
        if not self.progress_cb:
            return
        try:
            self.progress_cb(event_data)
        except Exception as e:
            print(f"Warning: Progress callback failed for {event_data.get('job_key', 'unknown')}: {e}")
    
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
                            self._safe_progress_callback({
                                "event": "downloaded",
                                "job_key": f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}",
                                "job_name": job.job_name,
                                "results_path": combined_path,
                            })
                            return True

            # If no ext url or failed download, leave a breadcrumb so a later pass can fetch from UI
            if out_ds_id:
                with open(os.path.join(results_dir, f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}_OUTPUT_DATASET_ID.txt"), "w", encoding="utf-8") as f:
                    f.write(str(out_ds_id))
                self._safe_progress_callback({
                    "event": "download_pending",
                    "job_key": f"t{self.temp_label}_{self.condition}_p{job.part_number:02d}",
                    "job_name": job.job_name,
                    "output_dataset_id": out_ds_id,
                })
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
