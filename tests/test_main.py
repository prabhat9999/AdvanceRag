import pytest

from main import parse_args


def test_accepts_question():
    args = parse_args(["What is agent memory?"])

    assert args.question == "What is agent memory?"


def test_requires_question():
    with pytest.raises(SystemExit) as error:
        parse_args([])

    assert error.value.code == 2


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_blank_question(question):
    with pytest.raises(SystemExit) as error:
        parse_args([question])

    assert error.value.code == 2