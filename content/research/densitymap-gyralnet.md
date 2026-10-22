---
title: "DensityMap-GyralNet"
date: 2024-07-01
repo: "https://github.com/HenryVarro666/DensityMap-GyralNet"
lang: Python
topics: [brain-imaging]
---
Extracts the gyral net — the network of gyral crests whose three-way junctions are 3-hinge gyri — from cortical surfaces. Reads VTK surface meshes with sulcal depth (or gradient-density maps, the “DensityMap” variant), isolates and thins gyral regions, grows a skeleton tree per region and repairs missing crest links along shortest paths; run on HCP surfaces.
