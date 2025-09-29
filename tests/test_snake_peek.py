from core import snake_peek


def test_peek_summary_basic(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text(
        '''"""Module docstring"""\nimport os\n# TODO: fix this\ndef foo():\n    pass\n''')

    summary = snake_peek.peek_summary(p)
    assert summary['path'].endswith('sample.py')
    assert 'os' in summary['imports']
    assert 'foo' in ''.join(summary['defs'])
    assert summary['todos']


def test_simple_score_returns_number(tmp_path):
    p = tmp_path / "sample2.py"
    p.write_text('''# no doc\nimport sys\n# TODO: a\n''')
    summ = snake_peek.peek_summary(p)
    score, breakdown = snake_peek.simple_score(summ)
    assert isinstance(score, float)
    assert isinstance(breakdown, dict)
