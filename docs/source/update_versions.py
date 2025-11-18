"""
This script automatically scans the gh-pages branch, finds our the generated
documentation numbers, sorts them, and creates the version_switcher.json file.
In addition, it creates the index.html file that redirects the user to the most
recent stable documentation version.
"""
# Imports
import os
import json
import re

# Constants
VERSION_NUMBER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
BASE_URL = "https://nmroulu.github.io/Spinguin"
BASE_FOLDER = "versionfiles"

def update_versions():
    # Obtain all folders from the root of gh-pages
    folders = [item for item in os.listdir(".") if os.path.isdir(item)]

    # Collect the release numbers from the folders
    latest_exists = False
    releases = []
    for folder in folders:
        if folder == "latest":
            latest_exists = True
        elif VERSION_NUMBER.match(folder):
            release = folder.split(".")
            releases.append(release)

    # Check that releases exist
    if len(releases) == 0:
        raise ValueError("Could not find any releases.")

    # Sort first by patch, then by minor, then by major
    releases = sorted(releases, key=lambda release: release[2], reverse=True)
    releases = sorted(releases, key=lambda release: release[1], reverse=True)
    releases = sorted(releases, key=lambda release: release[0], reverse=True)

    # Create a temporary directory
    os.makedirs(BASE_FOLDER)

    # Construct JSON
    json_data = []

    # Add latest to the JSON if the folder existed
    if latest_exists:
        json_data.append({
            "version": "latest",
            "url": f"{BASE_URL}/latest/"
        })

    # Add the releases to the JSON
    for release in releases:
        json_data.append({
            "version": f"{release[0]}.{release[1]}.{release[2]}",
            "url": f"{BASE_URL}/{release[0]}.{release[1]}.{release[2]}/"
        })

    # Create the version_switcher.json file
    with open(f"{BASE_FOLDER}/version_switcher.json", "w") as file:
        json.dump(json_data, file, indent=2)

    # Create the index.html file
    latest_release = f"{releases[0][0]}.{releases[0][1]}.{releases[0][2]}"
    html = \
f"""\
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url={latest_release}/">
</head>
<body></body>
</html>\
"""
    with open(f"{BASE_FOLDER}/index.html", "w") as file:
        file.write(html)

# Ensure that this script is not run when imported
if __name__ == "__main__":
    update_versions()