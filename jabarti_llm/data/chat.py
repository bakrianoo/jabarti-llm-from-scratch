"""
chat.py -- turning question/answer pairs into instruction-following examples
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_PLACEHOLDER = 0

DEFAULT_SYSTEM = {
    "ar": "أنت مساعد مفيد يجيب عن الأسئلة حول مصر.",
    "en": "You are a helpful assistant answering questions about Egypt.",
}

# Every character has a number (its Unicode code point); ord("م") == 0x0645.
# Arabic letters all live in one block of numbers: 0x0600 to 0x06FF.
ARABIC_LETTERS = range(0x0600, 0x0700)

def system_for(text):
    for char in text:
        if ord(char) in ARABIC_LETTERS:
            return DEFAULT_SYSTEM["ar"]

    return DEFAULT_SYSTEM["en"]

def format_chat(tokenizer, question, answer=None, system=None):

    """Build one conversation, and say which tokens the model is graded on.

    The layout uses the role tokens ch03 set aside:

        [SYS] system [USER] question [ASST] answer [EOS]

    Returns (input_ids, labels). labels is input_ids shifted by one, with
    every prompt position replaced by pad_id so the loss ignores it.

    That masking is the whole idea of instruction finetuning. Training on the
    prompt teaches the model to predict questions, which nobody wants -- it
    should only be graded on the reply it was supposed to produce.
    """

    prompt_ids = [tokenizer.BOS]
    if system:
        prompt_ids += [tokenizer.SYS] + tokenizer.encode(system)

    prompt_ids += [tokenizer.USER] + tokenizer.encode(question)
    prompt_ids += [tokenizer.ASST]

    if answer is None:
        return prompt_ids, None

    answer_ids = tokenizer.encode(answer) + [tokenizer.EOS]
    full_prompt = prompt_ids + answer_ids

    inputs = full_prompt[:-1]
    targets = full_prompt[1:]

    # ====
    # prompt_ids = [BOS, USER, q1, q2, ASST]     len = 5
    # answer_ids = [a1, a2, EOS]                 len = 3
    # full       = [BOS, USER, q1, q2, ASST, a1, a2, EOS]
    # ====
    # inputs  = full[:-1] = [BOS,  USER,  q1,  q2,   ASST, a1, a2 ]
    # targets = full[1:]  = [USER, q1,   q2,   ASST,  a1,   a2, EOS]

    first_graded = len(prompt_ids) - 1
    labels = [PAD_PLACEHOLDER] * first_graded + targets[first_graded:]

    return inputs, labels


class ChatDataset(Dataset):
    """Question/answer pairs as masked training examples.
    """

    def __init__(self, questions, answers, tokenizer,
                 config, languages=None, system=None):

        self.examples = []

        if languages is None:
            languages = [None] * len(questions)

        for question, answer, language in tqdm(
            zip(questions, answers, languages), total=len(questions),
            desc="tokenizing"
        ):
            if language is None:
                language = "en"

            row_system = system if system is not None else DEFAULT_SYSTEM[language]

            inputs, labels = format_chat(
                tokenizer=tokenizer, question=question, answer=answer, system=row_system
            )

            if len(inputs) > config.max_seq_len:
                continue

            self.examples.append(
                (inputs, labels)
            )

    @classmethod
    def from_parquet(cls, path, tokenizer, config, limit=None, **kwargs):
        frame = pd.read_parquet(path, columns=["question", "answer", "language"])
        if limit is not None:
            frame = frame.head(limit)

        return cls(
            questions=frame["question"].tolist(),
            answers=frame["answer"].tolist(), 
            languages=frame["language"].tolist(),

            tokenizer=tokenizer,
            config=config,

            **kwargs
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        inputs, labels = self.examples[index]
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
            
def chat_collate_fn(examples, pad_id=0):
    """Pad a batch of chat examples.
    """

    longest = max( len(example["input_ids"]) for example in examples)

    input_ids = torch.full(
                    (len(examples), longest), pad_id, dtype=torch.long
                )
    targets =   torch.full(
                    (len(examples), longest), pad_id, dtype=torch.long
                )

    for ix, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[ix, :length] = example["input_ids"]
        targets[ix, :length] = example["labels"]

    return {
        "input_ids": input_ids,
        "targets": targets,
    }

