from pydantic import BaseModel
from typing import Optional, Union
import json

class TaskDescription(BaseModel):
    description: str

class Task(BaseModel):
    '''A task is a unit of work that can be run asynchronously.'''
    task_id: int
    description: str
    result: Optional[str]
    status: str

    def __init__(self, description: str, task_id: int = None, result: str = None, status: str = "pending"):
        super().__init__(
            task_id=task_id, 
            description=description,
            status=status,
            result=result
            )

    def __str__(self) -> str:
        if self.result is None:
            return f"Task: {self.description} ({self.status})"
        else:
            return f"Task: {self.description} ({self.status})\n{self.result}"

    def __repr__(self) -> str:
        return self.model_dump_json(indent = 2)
        
    def toJson(self) -> str:
        return self.__repr__()
        
    def pending(self) -> None:
        self.status = "pending"

    def running(self) -> None:
        self.status = "running"

    def done(self, result: str = None) -> None:
        self.status = "done"
        if result is not None:
            self.result = result

    def get_status(self) -> str:
        return self.status
    
    def get_result(self) -> str:
        return self.result
    
class TaskList(BaseModel):
    tasks: Optional[list[Task]] = []

    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, key: int) -> Task:
        return self.tasks[key]
    
    def __setitem__(self, key: int, value: Task) -> None:
        self.tasks[key] = value

    def __delitem__(self, key: int) -> None:
        del self.tasks[key]

    def __iter__(self) -> iter:
        return iter(self.tasks)
    
    def __contains__(self, item: Task) -> bool:
        return item in self.tasks
    
    def append(self, item: Task) -> None:
        self.tasks.append(item)

    def extend(self, item: list[Task]) -> None:
        self.tasks.extend(item)