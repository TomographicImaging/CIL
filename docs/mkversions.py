#!/usr/bin/env python
import json
import re
from pathlib import Path
from packaging import version

baseurl = "/CIL/"
build = Path(__file__).parent / "build"
versions = [{
    "name": str(i.name),
    "version": re.search(
        "VERSION: '(.*)'",
        (i / "_static" / "documentation_options.js").read_text(),
        flags=re.M).group(1),
    "url": f"{baseurl}{i.name}/"
}
for i in build.glob("[a-zA-Z]*") if i.is_dir()]

tags = [v for v in versions if v["name"] == "v" + v["version"]]
try:
    stable = max(tags, key=lambda v: version.parse(v["version"]))
except ValueError:
    pass
else:
    versions += [{"name": "stable", "version": stable["version"], "url": stable["url"], "preferred": True}]

(build / "versions.json").write_text(json.dumps(versions))
