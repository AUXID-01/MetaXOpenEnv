import sys
sys.path.insert(0, r"e:\MetaXOpenEnv\negotiation-env")
from client.env_client import DummyEnvClient
from training.rollout import run_episode

client = DummyEnvClient()

def dummy_generate(prompt):
    return "<action_type>send_message</action_type>\n<text>This is my generated response.</text>"

trajectory = run_episode(client, dummy_generate, stage=1)
print(trajectory["prompts"][2])
