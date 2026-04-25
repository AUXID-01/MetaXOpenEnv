from fastapi import APIRouter, HTTPException

from api.schemas import ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse
from environment.env import NegotiationEnv


router = APIRouter()
_env = NegotiationEnv()


@router.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    obs = _env.reset(curriculum_stage=req.curriculum_stage)
    profile_id = _env._profile.get("id", "unknown") if _env._profile else "unknown"
    return ResetResponse(
        observation=obs.model_dump(),
        episode_id=obs.episode_id,
        profile_id=profile_id,
        curriculum_stage=req.curriculum_stage,
    )


@router.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    if not _env._episode_id:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    if req.episode_id != _env._episode_id:
        raise HTTPException(status_code=409, detail="episode_id does not match active episode.")

    obs, reward, done, info = _env.step(
        {
            "action_type": req.action_type,
            "text": req.text,
            "metadata": req.metadata,
        }
    )
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    if not _env._episode_id:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return StateResponse(state=_env.state(), episode_id=_env._episode_id)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
