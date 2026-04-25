class CurriculumScheduler:
    def __init__(self):
        pass
        
    def advance_if_ready(self, mean_reward_last_100: float) -> bool:
        """
        Evaluates whether the model's rolling average reward crosses the threshold to advance the curriculum.
        Stubbed to always return False while Person A and B work on stabilization.
        """
        return False
