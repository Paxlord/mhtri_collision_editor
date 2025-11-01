# MH Tri Collision Editor

A Blender extension for importing, editing, and exporting Monster Hunter Tri collision files.

## Description

This extension provides tools for working with .sch collision files from Monster Hunter Tri. It allows users to load collision meshes into Blender, modify them, and export them back to the game format.

The collision system uses a spatial grid structure where triangular polygons are assigned to grid cells based on their position. Each polygon contains geometric data and custom attributes that control collision behavior.

## Features

- Import .sch collision files as Blender meshes
- Import .bin collision archive files as multiple Blender meshes
- Export Blender meshes to .sch format
- Export Blender meshes to .bin archive format
- View and edit polygon attributes per face
- Configurable grid cell size on export
- Preserves polygon metadata including polyId, flags, and custom attributes

## Installation

Download the latest release zip file from https://github.com/Paxlord/mhtri_collision_editor/releases and install through Blender's extension manager. Go to Edit > Preferences > Get Extensions, click the dropdown menu in the top right, and select Install from Disk.

## Usage

Import and export functions are available in File > Import and File > Export menus under "MH Tri Collision (.sch)".

The collision panel appears in the 3D viewport sidebar under the MH Tri tab. Select a face in edit mode to view its collision attributes.

## Requirements

Blender 4.2.0 or higher.

## License

GPL-2.0-or-later

## Contact

For bug reports or feature requests, open an issue on the GitHub repository. You can also reach out on Discord at @pax_777
