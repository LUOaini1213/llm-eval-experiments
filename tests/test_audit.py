"""Dataset-audit checks on small synthetic datasets with planted problems."""
from evalab import audit as A


def _flags(examples):
    return {(i, c) for i, c, _ in A.check_examples(examples)}


def test_mcq_key_letter_text_mismatch():
    ex = [{"input": "Year?\nA) 1987\nB) 1990\nC) 1995\nD) 2000", "target": "B) 1987"}]
    fl = A.check_examples(ex)
    assert (0, "mcq_key_mismatch") in {(i, c) for i, c, _ in fl}
    assert "option A" in [d for _, c, d in fl if c == "mcq_key_mismatch"][0]


def test_mcq_key_missing_and_clean_item():
    ex = [{"input": "Year?\nA) 1997\nB) 2000\nC) 2003\nD) 2005", "target": "B) 2001"},
          {"input": "Year?\nA) 1997\nB) 2000", "target": "E) 1997"},
          {"input": "Who?\nA) Lee Kuan Yew\nB) Goh Chok Tong", "target": "A) Lee Kuan Yew"}]
    f = _flags(ex)
    assert (0, "mcq_key_missing") in f and (1, "mcq_key_missing") in f
    assert not any(i == 2 for i, _ in f)


def test_mcq_duplicate_and_all_of_the_above():
    ex = [{"input": "Q?\nA) Red\nB) red\nC) All of the above", "target": "A) Red"}]
    f = _flags(ex)
    assert (0, "mcq_duplicate_option") in f and (0, "ambiguity") in f


def test_tnf_checks():
    ex = [{"input": "Answer in TRUE or FALSE: Singapore has 5 regions", "target": "YES"},
          {"input": "Answer in TRUE or FALSE: Durians are usually banned on the MRT", "target": "TRUE"},
          {"input": "Answer in TRUE or FALSE: The Merlion is in Marina Bay", "target": "TRUE"}]
    f = _flags(ex)
    assert (0, "tnf_bad_target") in f
    assert (1, "ambiguity") in f
    assert not any(i == 2 for i, _ in f)


def test_instruction_words_do_not_trigger_ambiguity():
    # "only" in the answer-format instruction must not count as an absolute claim
    ex = [{"input": "Is the statement in single quotes TRUE or FALSE in the Singapore context? Respond with only "
                    "TRUE or FALSE. 'The CPF was set up in 1955.'", "target": "TRUE"}]
    assert (0, "ambiguity") not in _flags(ex)


def test_encoding_header_whitespace_and_time():
    ex = [{"input": "Is it �INLIS�?", "target": "TRUE"},
          {"input": "Term", "target": "Explanation"},
          {"input": "Context:   three   spaces", "target": " padded "},
          {"input": "Answer in TRUE or FALSE: As of 2023 the minimum occupation period is 5 years", "target": "TRUE"}]
    f = _flags(ex)
    assert (0, "encoding") in f and (1, "header_row") in f and (2, "whitespace") in f and (3, "time_sensitive") in f
    assert (1, "encoding") not in f


def test_duplicates_across_datasets():
    named = {"a": [{"input": "What is HDB?", "target": "A"}, {"input": "Which MRT line is purple?", "target": "NE"}],
             "b": [{"input": "What is HDB ?", "target": "A"}, {"input": "What is HDB?", "target": "B"},
                   {"input": "Which MRT line is coloured purple?", "target": "NE"}]}
    d = {(a, b): k for a, b, k, _ in A.duplicates(named, near=0.8)}
    assert d[("a#0", "b#0")] == "exact_duplicate"
    assert d[("a#0", "b#1")] == "same_input_different_target"
    assert d[("a#1", "b#2")] == "near_duplicate"
    assert ("a#0", "a#1") not in d


def test_metadata_name_checks():
    assert "misspelling" in A.metadata_check("singapore-pofma-statements-2023", {"name": "Singapore POMFA Statements"})
    assert "does not match" in A.metadata_check("singapore-public-housing", {"name": "Singapore Transport System"})
    assert A.metadata_check("singapore-transport-system", {"name": "Singapore Transport System"}) is None


def test_edit_distance():
    assert A.edit_distance("pofma", "pomfa") == 2
    assert A.edit_distance("", "abc") == 3
    assert A.edit_distance("same", "same") == 0
