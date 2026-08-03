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
        # Initialization
        title = "The Setup: A Strange Counting Game"
        lines = [
            "Imagine two blocks on a frictionless surface.",
            "A small mass m and a large mass M.",
            "We count every collision with the wall or each other."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_WALL = "#FFFFFF"
        COLOR_M = "#00FF00"
        COLOR_BIG_M = "#0000FF"
        COLOR_CLACKS = "#FF00FF"
        
        # Assets
        PATH_WALL = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        PATH_BLOCK = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"

        # === Animation for Lecture Line 1 ===
        # Imagine two blocks on a frictionless surface.
        self.lecture[0].set_color(WHITE)
        
        # Floor and Wall
        floor = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=COLOR_WALL)
        
        # Use Asset: wall.svg
        wall = SVGMobject(PATH_WALL).set_color(COLOR_WALL)
        self.place_in_area(wall, "A1", "F1", scale_factor=0.6)
        
        # Block m (small) - Use Asset: block.svg
        block_m = SVGMobject(PATH_BLOCK).set_color(COLOR_M)
        self.place_at_grid(block_m, "F2", scale_factor=0.4)

        # Block M (large) - Use Asset: block.svg
        block_M = SVGMobject(PATH_BLOCK).set_color(COLOR_BIG_M)
        self.place_at_grid(block_M, "F5", scale_factor=0.8)

        self.play(Create(floor), FadeIn(wall))
        self.play(FadeIn(block_m), FadeIn(block_M))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A small mass m and a large mass M.
        self.play(self.lecture[1].animate.set_color(COLOR_M))
        
        label_m = MathTex("m", color=COLOR_M)
        label_M = MathTex("M", color=COLOR_BIG_M)
        
        # Resolving Issues 34 & 36
        self.place_at_grid(label_m, "E2", scale_factor=1.2)
        self.place_at_grid(label_M, "E5", scale_factor=1.2)
        
        self.play(Write(label_m), Write(label_M))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We count every collision with the wall or each other.
        self.play(self.lecture[2].animate.set_color(COLOR_CLACKS))
        
        # Counter
        clacks_val = ValueTracker(0)
        counter_label = Text("Clacks:", font_size=24, color=COLOR_CLACKS)
        counter_num = DecimalNumber(0, num_decimal_places=0, color=COLOR_CLACKS)
        counter_num.add_updater(lambda d: d.set_value(clacks_val.get_value()))
        
        counter_group = VGroup(counter_label, counter_num).arrange(RIGHT, buff=0.2)
        # Resolving Issue 35
        self.place_at_grid(counter_group, "B5", scale_factor=1.1)
        
        self.play(FadeIn(counter_group))

        # Collision Sequence
        # 1. M moves left and hits m
        target_M_collision = self.grid["F2"] + RIGHT * 0.6
        self.play(block_M.animate.move_to(target_M_collision), run_time=1.5, rate_func=linear)
        
        # Collision 1 (with each other)
        clacks_val.set_value(1)
        
        # 2. m moves to wall and hits it
        target_m_wall = self.grid["F1"] + RIGHT * 0.4
        self.play(block_m.animate.move_to(target_m_wall), run_time=0.5, rate_func=linear)
        
        # Collision 2 (with the wall)
        clacks_val.set_value(2)
        
        # 3. m bounces back to collide with M again (implied end of count intro)
        self.play(block_m.animate.move_to(self.grid["F2"]), run_time=0.5, rate_func=linear)
        
        self.wait(2)
