"""
collate.py -- examples into batches
"""

import torch


def collate_fn(examples, pad_id=0):
    """Pad a list of variable-length chunks into one batch, then shift.
    
    Collate = gather separate pieces together and put them in order.

    The everyday example is a photocopier.
    If you print 3 copies of a 10-page document:

    Without collate:
        == 3 copies of page 1, then 3 of page 2, … then 3 of page 10 — grouped by page
    With collate:
        == pages 1→10, then pages 1→10 again, then a third time — 3 complete, ordered sets

    So "collate" means take loose individual items and assemble them into a proper, ordered group.

    Two things happen here, in this order:

    1. Pad to the longest chunk IN THIS BATCH, not to max_seq_len. A batch of
        short documents stays short, so no compute is spent on padding that a
        fixed width would have forced (ch02 Check-6).

    2. Shift by one. input_ids[t] is asked to predict targets[t], which is the
        token that actually followed it.

        Two chunks of different length arrive:

            position   0        1        2        3        4        5
            chunk A    [BOS]    Ramses   the      Second   was      [EOS]
            chunk B    [BOS]    He       ruled    Egypt    [EOS]

        Step 1 -- pad the short one out to 6, the longest in this batch:

            row A      [BOS]    Ramses   the      Second   was      [EOS]
            row B      [BOS]    He       ruled    Egypt    [EOS]    [PAD]

        Step 2 -- slice. input drops the last column, target drops the first:

            input  A   [BOS]    Ramses   the      Second   was
            target A            Ramses   the      Second   was      [EOS]

            input  B   [BOS]    He       ruled    Egypt    [EOS]
            target B            He       ruled    Egypt    [EOS]    [PAD]

        Read a column: given everything up to "Ramses", predict "the"; given
        everything up to "the", predict "Second". One chunk of N tokens is
        therefore N-1 training examples, not one.

        Row B is the padded case. Its last column asks the model to predict
        [PAD] after [EOS] -- a question with no right answer. That is exactly
        the position ignore_index throws away, so it costs no loss and no
        gradient. Only real tokens ever teach the model anything.

    Padding lands in targets as pad_id, and the loss ignores those positions
    (ignore_index in GPT.forward). Note what is deliberately NOT here: an
    attention padding mask. Padding only ever sits at the END of a sequence,
    and the causal mask already stops every real token from looking forward,
    so no real token can attend to a pad. A padding mask would be code that
    changes nothing.
    """

    chunks = [ example["input_ids"] for example in examples ]
    longest = max( len(chunk) for chunk in chunks)

    padded = torch.full(
        (len(chunks), longest), pad_id, dtype=torch.long
    )
    for ix, chunk in enumerate(chunks):
        padded[ix, :len(chunk)] = chunk

    return {
        "input_ids": padded[: , :-1],
        "targets": padded[:, 1:]
    }

