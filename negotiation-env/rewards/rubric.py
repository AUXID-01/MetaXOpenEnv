"""Reward rubric wrapper used by the environment runtime."""

from reward import Rubric as _CoreRubric


class Rubric:
    """Compatibility wrapper so tests can monkeypatch rewards.rubric.Rubric."""

    def __init__(self, curriculum_stage: int = 3):
        self._impl = _CoreRubric(curriculum_stage=curriculum_stage)

    def compose(self, state_before, state_after, action, episode_done) -> tuple[float, dict]:
        return self._impl.compose(
            state_before=state_before,
            state_after=state_after,
            action=action,
            episode_done=episode_done,
        )
