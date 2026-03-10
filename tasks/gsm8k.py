"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
import json
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# ---------------------------------------------------------------------------
# Reward function registry — each function: (conversation, response) -> float
# ---------------------------------------------------------------------------

def reward_correctness(conversation, assistant_response):
    """Original binary correctness reward: 1.0 if exact match, 0.0 otherwise."""
    assistant_message = conversation['messages'][-1]
    last_text_part = assistant_message['content'][-1]['text']
    ref_num = extract_answer(last_text_part)
    pred_num = extract_answer(assistant_response)
    return float(int(pred_num == ref_num))

def reward_format_compliance(conversation, assistant_response):
    """Reward A: 1.0 if response contains #### <number>, 0.0 otherwise."""
    match = GSM_RE.search(assistant_response)
    return 1.0 if match else 0.0

def reward_numeric_proximity(conversation, assistant_response):
    """Reward B: partial credit based on distance to gold answer."""
    assistant_message = conversation['messages'][-1]
    last_text_part = assistant_message['content'][-1]['text']
    ref_match = GSM_RE.search(last_text_part)
    if not ref_match:
        return 0.0
    ref_str = ref_match.group(1).strip().replace(",", "")
    pred_match = GSM_RE.search(assistant_response)
    if not pred_match:
        return 0.0
    pred_str = pred_match.group(1).strip().replace(",", "")
    try:
        ref_num = float(ref_str)
        pred_num = float(pred_str)
    except ValueError:
        return 0.0
    distance = abs(pred_num - ref_num)
    denominator = abs(ref_num) + 1.0
    return max(0.0, 1.0 - distance / denominator)

# Stubs for post-P3 rewards
def reward_c(conversation, assistant_response):
    """Reward C: TBD — fill after P3 error analysis."""
    raise NotImplementedError("Reward C not yet designed — requires P3 deliverable")

def reward_d(conversation, assistant_response):
    """Reward D: TBD — fill after P3 error analysis."""
    raise NotImplementedError("Reward D not yet designed — requires P3 deliverable")

REWARD_REGISTRY = {
    "correctness": reward_correctness,
    "format_compliance": reward_format_compliance,
    "numeric_proximity": reward_numeric_proximity,
    "reward_c": reward_c,
    "reward_d": reward_d,
}

def load_reward_config(path):
    """Load a JSON reward config file. Returns list of reward names to use."""
    with open(path) as f:
        config = json.load(f)
    names = config.get("rewards", ["correctness"])
    for name in names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward '{name}'. Available: {list(REWARD_REGISTRY.keys())}")
    return names


class GSM8K(Task):

    def __init__(self, subset, split, reward_names=None, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        # Reward dispatch: list of reward component names to sum
        self.reward_names = reward_names or ["correctness"]

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. Dispatches through self.reward_names and sums components.
        Returns (total_reward, per_component_dict).
        """
        components = {}
        for name in self.reward_names:
            components[name] = REWARD_REGISTRY[name](conversation, assistant_response)
        total = sum(components.values())
        return total, components
