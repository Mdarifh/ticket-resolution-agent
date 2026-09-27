"""Project endpoints: the groups that runs belong to."""

from fastapi import APIRouter, Depends, status

from qa_agent.api.dependencies import get_run_manager
from qa_agent.api.schemas import ProjectCreate, ProjectOut
from qa_agent.services.run_manager import RunManager

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
def list_projects(manager: RunManager = Depends(get_run_manager)) -> list[ProjectOut]:
    return [ProjectOut(**project) for project in manager.list_projects()]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProjectOut)
def create_project(body: ProjectCreate, manager: RunManager = Depends(get_run_manager)) -> ProjectOut:
    return ProjectOut(**manager.create_project(body.name, body.description))
