# Spleen Sweeper Solver (dota2 minesweeper)

## Features:
* game field auto-detection
* full automatic solving
* multiple games in single run
* shutdown at any time
* game end detection
* pause game to save more time
* mana and clocks collection with priority
* Kez's skill usage

## Demonstration:
![demonstration](misc/demonstration.gif)

## Not implemented:
* skip dialogues
* auto start (you have to press S / Space to start)
* different screen resolution (only 1920x1080, you may change in dota2 setting)
* different screen scaling (you may change in script)
* can't recognize 7/8 cells, because I haven't seen any of them

## Usage
* check and validate consts for your system
* python3 solver.py - to win this game
* press "S" or "Space" to start game. One press per stage
* (optionally) python3 train.py - if you want to add more train/val samples and train new recognizer (cpu train, about 1 min)
