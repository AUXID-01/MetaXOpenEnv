from contracts import CURRICULUM_STAGES


class CurriculumScheduler:
    def __init__(self, initial_stage: int = 1, max_stage: int = 4):
        self.current_stage = initial_stage
        self.max_stage = max_stage

    def advance_if_ready(self, mean_reward_last_100: float) -> bool:
        """Advance stage once threshold is crossed for current stage."""
        if self.current_stage >= self.max_stage:
            return False

        threshold = CURRICULUM_STAGES.get(self.current_stage, {}).get("advance_on_reward")
        if threshold is None:
            return False

        if mean_reward_last_100 >= threshold:
            self.current_stage += 1
            return True

        return False
