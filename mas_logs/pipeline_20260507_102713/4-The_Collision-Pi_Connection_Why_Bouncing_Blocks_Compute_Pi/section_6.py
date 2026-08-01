from manim import *
import numpy as np
import os
import pathlib

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define layout
        title = "Conclusion: Physics as a Computer"
        lines = [
            "This simple system acts as a mechanical computer for Pi.",
            "It reveals a deep link between dynamics and geometry.",
            "Mathematical beauty emerges from the simplest physical laws."
        ]
        self.setup_layout(title, lines)

        # Colors for lines
        COLOR_1 = "#ADD8E6" # Light Blue
        COLOR_2 = "#00FF00" # Green
        COLOR_3 = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Setup visual elements: Wall, Blocks, Counter, Computer Icon
        wall = Line(
            self.grid["B1"] + UP * 0.5, 
            self.grid["E1"] + DOWN * 0.5, 
            color=WHITE, 
            stroke_width=6
        )
        
        small_block = Square(side_length=0.6, fill_opacity=0.8, color=COLOR_1, fill_color=COLOR_1)
        self.place_at_grid(small_block, "C2")
        
        large_block = Square(side_length=1.2, fill_opacity=0.8, color=BLUE_E, fill_color=BLUE_E)
        self.place_at_grid(large_block, "C4")
        
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/computer.svg]
        computer_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_icon, "C5", scale_factor=0.8)
        computer_icon.set_color(COLOR_1)

        counter_label = Text("Collisions:", font_size=24, color=WHITE)
        self.place_at_grid(counter_label, "A2", scale_factor=0.8) # Issue 44: Scale adjustment
        
        count_tracker = ValueTracker(0)
        
        # FIXED: Replace Integer(0) with Text("0") to avoid LaTeX dependency error ('latex' not found)
        counter_val = Text("0", font_size=36, color=WHITE)
        self.place_at_grid(counter_val, "A3", scale_factor=0.8) # Issue 42: Grid and Scale adjustment
        
        # Update counter based on tracker using Text and become()
        # Scale and position are maintained during replacement
        counter_val.add_updater(lambda m: m.become(
            Text(str(int(count_tracker.get_value())), font_size=36, color=WHITE).scale(0.8).move_to(self.grid["A3"])
        ))

        self.add(wall, small_block, large_block, counter_label, counter_val, computer_icon)

        # Animation: Fast-forward collision counter
        small_origin = small_block.get_center().copy()
        large_origin = large_block.get_center().copy()
        shake_small = lambda m: m.move_to(small_origin + RIGHT * 0.05 * np.sin(self.renderer.time * 50))
        shake_large = lambda m: m.move_to(large_origin + LEFT * 0.02 * np.sin(self.renderer.time * 10))
        
        small_block.add_updater(shake_small)
        large_block.add_updater(shake_large)
        
        self.play(
            count_tracker.animate.set_value(3141592),
            run_time=4,
            rate_func=linear
        )
        
        small_block.remove_updater(shake_small)
        large_block.remove_updater(shake_large)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Highlight counter in green
        self.play(
            counter_val.animate.set_color(COLOR_2).scale(1.2),
            counter_label.animate.set_color(COLOR_2)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Fade out interaction elements
        fade_group = VGroup(wall, small_block, large_block, counter_label, counter_val, computer_icon)
        
        # FIXED: Replace MathTex(r"\pi") with Text("π") to avoid LaTeX dependency error
        pi_symbol = Text("π", font_size=180, color=COLOR_3)
        self.place_in_area(pi_symbol, "B2", "E5", scale_factor=0.7) # Issue 43: Scale adjustment

        self.play(
            FadeOut(fade_group),
            FadeIn(pi_symbol, shift=UP * 0.5)
        )
        
        self.play(pi_symbol.animate.scale(1.1), run_time=1.5, rate_func=there_and_back)
        self.wait(3)
