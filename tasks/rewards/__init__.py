from tasks.rewards.correctness import reward_correctness
from tasks.rewards.format_compliance import reward_format_compliance
from tasks.rewards.numeric_proximity import reward_numeric_proximity
from tasks.rewards.reward_c import reward_c
from tasks.rewards.reward_d import reward_d

REWARD_REGISTRY = {
    "correctness": reward_correctness,
    "format_compliance": reward_format_compliance,
    "numeric_proximity": reward_numeric_proximity,
    "reward_c": reward_c,
    "reward_d": reward_d,
}
