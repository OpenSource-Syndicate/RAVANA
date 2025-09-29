from core.snake_indexer import build_index, load_index


def test_build_and_load_index(tmp_path):
    # create a couple of files
    a = tmp_path / 'a.py'
    a.write_text(
        '''"""A module"""\nimport os\n# TODO: test\ndef f():\n    pass\n''')
    b = tmp_path / 'b.py'
    b.write_text('''# no doc\nimport sys\n''')

    index_file = tmp_path / '.snake_index.json'
    entries = build_index(tmp_path, index_file=index_file, max_files=20)
    assert isinstance(entries, list)
    # persisted index exists
    loaded = load_index(index_file)
    assert isinstance(loaded, list)
    assert len(loaded) >= 1
