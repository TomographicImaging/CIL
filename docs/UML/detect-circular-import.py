#!/usr/bin/env python3

# This script uses pyreverse to identify circular package imports in CIL.

# To generate packages.plain, you will need to run the command as follows from the terminal:
# `pyreverse -A -k -o plain ..\..\Wrappers\Python\cil\`

import os

fname = "output/overall-CIL-packages.plain"
if not os.path.exists(fname):
    raise IOError(fname + " does not exists. Please make sure you run `./extract-UML.sh` first.")

# A dictionary containing the edges
links = {}

# Open the file
f = open(fname, 'r')

# Process every line of the file
for line in f:

    # Split the line into words
    words = line.split(' ') 

    # Only care about edges and drop the rest
    if words[0] == "edge":

        # Extract the two packages of that edge
        package_1 = words[1].replace('"', '')
        package_2 = words[2].replace('"', '')

        if package_1 not in links:
            links[package_1] = []

        # Add the link
        links[package_1].append(package_2)

# Exhaustive search of circular
count = 0
for key1 in links:
    for key2 in links[key1]:
        # We know that the link key1 -> key2 exists.
        # We must now ensure that the link key2 -> key1 DOES NOT exist.
        # If it does, then there is a circular package import.
        if key2 in links:
            for key in links[key2]:
                if key == key1:
                    count += 1
                    print(count, "- Circular package import:", key1, key2)