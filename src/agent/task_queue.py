"""
Decoupled Background Task Queue Engine.
Manages background worker threads for strategy backtesting, sector analysis,
and asynchronous order slicing without blocking the main trading loop or API server.
"""

import logging
import queue
import threading
import time
import uuid
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AsyncTask:
    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: str = "queued"  # 'queued' | 'running' | 'completed' | 'failed'
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

class TaskQueueEngine:
    """
    Asynchronous background task queue runner.
    Decouples heavy AI / optimization tasks from REST API endpoints.
    """
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.tasks: Dict[str, AsyncTask] = {}
        self.workers: list = []
        self.running = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start worker threads."""
        if self.running:
            return
            
        self.running = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i+1}", daemon=True)
            t.start()
            self.workers.append(t)
        logger.info(f"TaskQueueEngine started with {self.num_workers} worker threads")
        
    def stop(self):
        """Stop worker threads."""
        self.running = False
        logger.info("TaskQueueEngine stopping...")
        
    def submit(self, name: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit a background task for execution.
        
        Args:
            name: Human-readable task name
            func: Callable function
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            task_id string
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = AsyncTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        with self._lock:
            self.tasks[task_id] = task
            
        self.task_queue.put(task)
        logger.info(f"Enqueued task [{name}] with ID {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status and result of a submitted task."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
                
            return {
                'task_id': task.task_id,
                'name': task.name,
                'status': task.status,
                'created_at': task.created_at.isoformat(),
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error': task.error,
                'result': task.result if task.status == 'completed' else None
            }

    def _worker_loop(self):
        """Worker thread execution loop."""
        while self.running:
            try:
                task: AsyncTask = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            task.status = "running"
            logger.info(f"Worker starting task [{task.name}] ({task.task_id})")
            
            try:
                res = task.func(*task.args, **task.kwargs)
                task.result = res
                task.status = "completed"
                logger.info(f"Worker completed task [{task.name}] ({task.task_id}) successfully")
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                logger.error(f"Worker failed task [{task.name}] ({task.task_id}): {e}")
            finally:
                task.completed_at = datetime.utcnow()
                self.task_queue.task_done()
