#!/usr/bin/env python
import json
import re
from os import getenv
from pathlib import Path
from packaging import version

baseurl = f'/{getenv("GITHUB_REPOSITORY", "").split("/", 1)[-1]}/'.replace("//", "/")
build = Path(__file__).parent / "build"
versions = [{
    "name": i.name,
    "version": re.search(
        "VERSION: '(.*)'",
        (i / "_static" / "documentation_options.js").read_text(),
        flags=re.M).group(1),
    "url": f"{baseurl}{i.name}/"
}
for i in build.glob("[a-zA-Z0-9]*") if i.is_dir() if i.name != "assets"]

tags = [v for v in versions if v["name"] == "v" + v["version"]]
try:
    stable = max(tags, key=lambda v: version.parse(v["version"]))
except ValueError:
    pass
else:
    versions += [{"name": "stable", "version": stable["version"], "url": stable["url"], "preferred": True}]

versions.sort(key=lambda v: (version.parse(v["version"]), v["name"]), reverse=True)
(build / "versions.json").write_text(json.dumps(versions))
