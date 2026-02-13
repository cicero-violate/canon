./target/debug/canon ingest --src ./canon --semantic-out /tmp/canon.semantic.json
./target/debug/canon materialize canon.semantic.json --layout canon.semantic.layout.json /tmp/canon_replicated
