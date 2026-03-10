"""Original binary correctness reward: 1.0 if exact match, 0.0 otherwise."""

from tasks.rewards.gsm8k_utils import extract_answer


def reward_correctness(conversation, assistant_response):
    assistant_message = conversation['messages'][-1]
    last_text_part = assistant_message['content'][-1]['text']
    ref_num = extract_answer(last_text_part)
    pred_num = extract_answer(assistant_response)
    return float(int(pred_num == ref_num))
