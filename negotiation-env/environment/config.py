# environment/config.py
"""
HOLDS: all environment constants.
RUNS: imported by env.py, adversary.py, rewards/rubric.py.
CONNECTS TO: env.py, adversary.py, rewards/rubric.py.
"""

# Episode length
MAX_TURNS = 15

# Global thresholds (can be overridden by profiles)
DEFAULT_ANGER_THRESHOLD = 8.0
SUCCESS_ANGER_THRESHOLD = 3.0
SUCCESS_TRUST_THRESHOLD = 7.0

# Trust reveal thresholds
TRUST_REVEAL_THRESHOLD = 5.0

# System prompts / constraints
CLASSIFIER_NOISE = False
CURRICULUM_STAGE = 1
