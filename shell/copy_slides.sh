cat ./docs/slides/manifest.txt | xargs -I % cp -r ./docs/slides/%/%.pdf ./docs/_build/html/