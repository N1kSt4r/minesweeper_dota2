import os
import cv2
import time
import torch
import numpy as np
import pyscreenshot as ImageGrab

from threading import Thread

from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Listener, Key, Controller as KeyController

USE_PAUSE = True
SHIFTS = [-1, 0, 1]
BORDER_COLOR = [83, 65, 43]
MAX_LEFT_CLICKS_SINGLE_STEP = 2
# 1 - optimal for bonuses
# 2/3 are optimal for user time / bonuses
# 100500 - unlimited. Optimal for user time

TRIES_SLEEP = 0.123
TRIES_NUMBER = 10
BORDER_WIDTH = 4
CELL_WIDTH = 36
SCREEN_SCALE = 1


def pairwise_L1_distance(left, right):
    left = np.array(left)[:, None] # n 1 d
    right = np.array(right)[None]  # 1 m d
    distances = np.max(np.abs(left - right), axis=-1) # L1 distance
    return distances


class Solver(Listener):
    def __init__(self):
        super().__init__(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.miss_count = 0
        self.terminated = False

        self.mouse = MouseController()
        self.keyboard = KeyController()

        self.recognizer = torch.load('recognizer.pth', weights_only=False).eval()
        self.solve_thread = None

        print('Use "q"/"esc" to stop, "s"/"space" to start')

    def _on_release(self, key):
        if key == Key.esc or getattr(key, 'char', None) == 'q':
            return False

    def __exit__(self, type, value, traceback):
        self.join() 
        super().__exit__(type, value, traceback)

    def _on_press(self, key):
        char = getattr(key, 'char', None)
        if char == 'q' or key == Key.esc:
            self.terminated = True
        elif char == 's' or key == Key.space:
            if self.solve_thread is None:
                self.solve_thread = Thread(target=self.solve, daemon=True)
                self.solve_thread.start()
            else:
                try:
                    self.solve_thread.join(timeout=0.5)
                    self.solve_thread = None
                    self._on_press(key)
                except:
                    print('Can\'t join the thread yet')
        elif char == 'z' or key == Key.f9:
            pass
        else:
            self.miss_count += 1
            if self.miss_count > 5:
                print('Use "q"/"esc" to stop, "s"/"space" to start')

    def solve(self):
        frame = np.array(ImageGrab.grab())
        self.find_borders()

        self.start_game()

        while not self.terminated:
            self.mouse.position = 0, 0
            self.press_key(Key.f9, press_time=1)
            self.press_key(Key.f9)
            field = self.get_field()
            text_field = self.get_text_field(field)

            line_field = ''.join(text_field)
            g_number = line_field.count('g')
            if g_number > 10:
                break
            elif g_number > 0:
                continue

            assert 'g' not in line_field

            text_field = self.add_margin(text_field)
            to_left, to_right = self.find_naive(text_field)
            to_left_hard, to_right_hard, maybe_cell = self.find_brute_force(text_field)
            to_right.update(to_right_hard)
            to_left += list(to_left_hard)

            if to_left:
                if 'c' in line_field:
                    clock_cells = self.find_char(text_field, 'c')
                    if clock_cells:
                        to_left = self.trunk_nearest(clock_cells, to_left)
                elif 'm' in line_field:
                    mana_cells = self.find_char(text_field, 'm')
                    if mana_cells:
                        to_left = self.trunk_nearest(mana_cells, to_left)

                to_left = list(to_left)[:MAX_LEFT_CLICKS_SINGLE_STEP]

            if to_left or to_right: 
                self.process_clicks(to_left, to_right, shift=-1)
            elif maybe_cell:
                self.shoot(maybe_cell, shift=-1)

    def find_borders(self):
        for i in range(TRIES_NUMBER):
            frame = np.array(ImageGrab.grab())
            mask = np.all(frame == BORDER_COLOR, axis=-1)

            coords = np.vstack(np.where(mask)).T

            # filter outliers
            distances = pairwise_L1_distance(coords, coords)
            distances = distances + np.eye(len(distances)) * np.max(distances) # remove zero diag
            coords = coords[np.min(distances, axis=0) < 5]

            begin = np.min(coords, axis=0) + BORDER_WIDTH
            end = np.max(coords, axis=0) + 1 - BORDER_WIDTH

            frame = frame[begin[0]: end[0], begin[1]: end[1]]
            if frame.shape[0] % CELL_WIDTH != 0 or frame.shape[1] % CELL_WIDTH != 0:
                time.sleep(TRIES_SLEEP)
                continue

            self.zero = begin
            self.borders = begin, end

            return
        raise RuntimeError("check")

    def cell2pixel(self, cell, shift=0):
        pixel = np.flip(self.zero) + CELL_WIDTH // 2 + (np.flip(cell) + shift) * CELL_WIDTH
        return tuple(pixel // SCREEN_SCALE)

    def process_clicks(self, to_left, to_right, shift=0):
        for cell in to_left:
            self.mouse.position = self.cell2pixel(cell, shift=shift)
            time.sleep(0.020)
            self.mouse.click(Button.left, 1)
            time.sleep(0.005)
        for cell in to_right:
            self.mouse.position = self.cell2pixel(cell, shift=shift)
            time.sleep(0.015)
            self.mouse.click(Button.right, 1)
            time.sleep(0.005)

    def press_key(self, key, press_time=0.015):
        if key != Key.f9 or USE_PAUSE:
            self.keyboard.press(key)
            time.sleep(press_time)
            self.keyboard.release(key)
            time.sleep(0.050)
        else:
            time.sleep(press_time)

    def shoot(self, cell, shift=0):
        self.press_key('1')
        self.process_clicks([cell], [], shift=shift)
        self.press_key(Key.f9, press_time=2.5)
        self.press_key(Key.f9)

    def start_game(self):
        frame = self.get_field()
        text_field = self.get_text_field(frame)
        center = len(text_field) // 2, len(text_field[0]) //2
        self.mouse.position = self.cell2pixel(center)
        self.process_clicks([center], [])

    def get_field(self):
        self.mouse.position = 0, 0
        time.sleep(0.050)

        borders = np.hstack(self.borders)
        borders = tuple(borders[[1, 0, 3, 2]])
        return np.array(ImageGrab.grab(borders))

    def add_margin(self, field):
        margin = list(' ' * (len(field[0]) + 2))
        return [margin] + [list(f' {line} ') for line in field] + [margin]

    def get_text_field(self, field):
        symbols = ".cmg *123456"
        tensor = torch.as_tensor(field, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            predict = torch.argmax(self.recognizer(tensor)[0], 0).tolist()

        text_field = []
        for row in predict:
            str_row = []
            for idx in row:
                str_row.append(symbols[idx])
            text_field.append(''.join(str_row))
        return text_field

    def is_unknown(self, char):
        return char in ".cm"

    def trunk_nearest(self, cells, candidates, count=100500):
        candidates = np.array(list(candidates))
        distances = np.min(pairwise_L1_distance(candidates, cells), axis=-1)
        idxs = np.argsort(distances)[:count]
        return candidates[idxs].tolist()

    def find_char(self, field, char):
        cells = []
        for i in range(1, len(field) - 1):
            for j in range(1, len(field[0]) - 1):
                if field[i][j] == char:
                    cells.append((i, j))
        return cells

    def find_naive(self, field):
        to_left, to_right = set(), set()

        for i in range(1, len(field) - 1):
            for j in range(1, len(field[0]) - 1):
                if '1' <= field[i][j] <= '8':
                    mines_max = int(field[i][j])
                    mines_count = 0
                    closed = []

                    for shift_i in SHIFTS:
                        for shift_j in SHIFTS:
                            if self.is_unknown(field[i + shift_i][j + shift_j]):
                                closed.append((i + shift_i, j + shift_j))
                            elif field[i + shift_i][j + shift_j] == '*':
                                mines_count += 1

                    if mines_count == mines_max and closed:
                        to_left.add((i, j, len(closed)))
                    elif mines_count + len(closed) == mines_max:
                        to_right.update(set(closed))

        to_left = sorted(to_left, key=lambda x: -x[-1])
        to_left = [cell[:2] for cell in to_left]
        return to_left, to_right

    def find_brute_force(self, field):
        border, to_check = self.find_stoke(field)
        count = [0] * len(border)

        self.succes_count = 0
        self.permute_stroke(field, to_check, border, count, 0)

        min_left_count = 1000
        min_left_idx = 0

        left_click = []
        right_click = []
        for i in range(len(count)):
            if count[i] < min_left_count:
                min_left_count = count[i]
                min_left_idx = i
            if count[i] == 0:
                left_click.append(border[i])
            elif count[i] == self.succes_count:
                right_click.append(border[i])

        maybe_cell = border[min_left_idx] if min_left_idx < len(border) else None

        return left_click, right_click, maybe_cell

    def is_stroke(self, field, to_check, i, j):
        flag = False
        if self.is_unknown(field[i][j]):
            for shift_i in SHIFTS:
                for shift_j in SHIFTS:
                    if field[i + shift_i][j + shift_j].isnumeric():
                        to_check.add((i + shift_i, j + shift_j))
                        flag = True
        return flag

    def find_stoke(self, field):
        to_check = set()
        border = set()

        for i in range(1, len(field) - 1):
            for j in range(1, len(field[0]) - 1):
                if self.is_stroke(field, to_check, i, j):
                    border.add((i, j))

        return sorted(border), sorted(to_check)

    def permute_stroke(self, field, to_check, border, count, idx):
        if idx == len(border):
            if self.check_field(field, to_check):
                self.succes_count += 1

                for idx in range(len(border)):
                    i, j = border[idx]
                    if field[i][j] == '*':
                        count[idx] += 1
            return

        i, j = border[idx]
        old_value = field[i][j]
        field[i][j] = '*'
        if self.check_cell(field, i, j):
            self.permute_stroke(field, to_check, border, count, idx + 1)
        field[i][j] = ' '
        if self.check_cell(field, i, j):
            self.permute_stroke(field, to_check, border, count, idx + 1)
        field[i][j] = old_value

    def check_field(self, field, to_check):
        for cell in to_check:
            i, j = cell
            mines_count = 0
            mines_max = int(field[i][j])

            for shift_i in SHIFTS:
                for shift_j in SHIFTS:
                    if field[i + shift_i][j + shift_j] == '*':
                        mines_count += 1

            if mines_count != mines_max:
                return False
        return True

    def check_cell(self, field, i, j):
        if field[i][j] in '* ':
            for shift_i in SHIFTS:
                for shift_j in SHIFTS:
                    new_i, new_j = i + shift_i, j + shift_j
                    if field[new_i][new_j].isnumeric() \
                            and not self.check_cell(field, new_i, new_j):
                        return False
            return True
        else:
            mines_max = int(field[i][j])
            mines_count = 0
            closed_count = 0

            for shift_i in SHIFTS:
                for shift_j in SHIFTS:
                    if self.is_unknown(field[i + shift_i][j + shift_j]):
                        closed_count += 1
                    elif field[i + shift_i][j + shift_j] == '*':
                        mines_count += 1

            if mines_count + closed_count < mines_max:
                return False
            if mines_count > mines_max:
                return False
            return True

# Collect events until released
with Solver() as solver:
    pass
