from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout(
            "The Strange Setup: A Counting Mystery", 
            [
                'Imagine a frictionless floor with a wall on the left.', 
                'A small block and a massive block sit nearby.', 
                'The large block slides in, starting a chain of collisions.'
            ]
        )
        
        # Define Colors
        COLOR_FLOOR = "#FFFFFF"
        COLOR_SMALL = "#00FF00"
        COLOR_LARGE = "#0000FF"
        COLOR_WALL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_FLOOR))
        
        # Floor line - spanning bottom row
        floor = Line(self.grid["F1"] + LEFT*0.5 + DOWN*0.5, self.grid["F6"] + RIGHT*0.5 + DOWN*0.5, color=COLOR_FLOOR)
        
        # Wall asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/wall.svg
        wall = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wall.svg")
        wall.set_color(COLOR_WALL)
        self.place_in_area(wall, "A1", "F1", scale_factor=1.0)
        
        self.play(Create(floor), FadeIn(wall))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_SMALL)
        )
        
        # Small block (m) - Green SVG: /mmfs1/data/home/jthen/Code2Video/assets/icon/block.svg
        small_block = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/block.svg")
        small_block.set_color(COLOR_SMALL)
        self.place_at_grid(small_block, "F2", scale_factor=0.6)
        
        small_label = Text("m", slant=ITALIC, color=WHITE)
        self.place_at_grid(small_label, 'E2', scale_factor=0.8) # Fix from Issue 29
        
        # Large block (M) - Blue SVG: /mmfs1/data/home/jthen/Code2Video/assets/icon/block.svg
        large_block = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/block.svg")
        large_block.set_color(COLOR_LARGE)
        self.place_at_grid(large_block, 'F5', scale_factor=1.3) # Fix from Issue 31
        
        large_label = Text("M", slant=ITALIC, color=WHITE)
        self.place_at_grid(large_label, 'D5', scale_factor=0.9) # Fix from Issue 30
        
        self.play(FadeIn(small_block), FadeIn(small_label))
        self.play(FadeIn(large_block), FadeIn(large_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LARGE)
        )
        
        # 1. Large block slides in (M slides left)
        self.play(
            large_block.animate.shift(LEFT * 2.2),
            large_label.animate.shift(LEFT * 2.2),
            run_time=2,
            rate_func=linear
        )
        
        # 2. Collision 1: M hits m
        # Use collision point calculation relative to current positions
        collision_pt1 = small_block.get_right()
        self.play(Flash(collision_pt1, color=WHITE, flash_radius=0.4))
        
        # m flies left towards wall, M slows (simulated)
        self.play(
            small_block.animate.shift(LEFT * 0.8),
            small_label.animate.shift(LEFT * 0.8),
            large_block.animate.shift(LEFT * 0.2),
            large_label.animate.shift(LEFT * 0.2),
            run_time=0.6,
            rate_func=linear
        )
        
        # 3. Collision 2: m hits Wall
        collision_pt_wall = wall.get_right()
        self.play(Flash(collision_pt_wall, color=WHITE, flash_radius=0.4))
        
        # m bounces back towards M
        self.play(
            small_block.animate.shift(RIGHT * 0.4),
            small_label.animate.shift(RIGHT * 0.4),
            run_time=0.4,
            rate_func=linear
        )
        
        self.wait(2)
