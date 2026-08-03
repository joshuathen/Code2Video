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
        self.setup_layout("The Mysterious Setup", [
            "Imagine two blocks between a wall and an observer.",
            "A massive block slides toward a smaller one.",
            "We count every collision until they drift away."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in a white wall (#FFFFFF) and a small green block (#00FF00) sitting next to it.
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Integration of assets and moving blocks to row F (Issue 20, 28)
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color="#FFFFFF")
        # Scale factor 5.0 ensures the wall covers the vertical span of the grid (2.2 to -2.8)
        self.place_in_area(wall, "A1", "F1", scale_factor=5.0)
        wall.shift(LEFT * 0.4)
        
        small_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#00FF00")
        self.place_at_grid(small_block, "F2", scale_factor=0.6)
        
        ground = Line(self.grid["F1"] + LEFT*0.6 + DOWN*0.5, self.grid["F6"] + RIGHT*0.5 + DOWN*0.5, color=WHITE)
        small_block.shift(DOWN * 0.1)

        self.play(Create(wall), Create(ground))
        self.play(FadeIn(small_block))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A massive blue block (#0000FF) slides in from the right toward the small block.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#0000FF")
        )
        
        large_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#0000FF")
        # Scaling down large block and moving to row F (Issue 29, 28)
        self.place_at_grid(large_block, "F6", scale_factor=0.8)
        large_block.shift(DOWN * 0.05) 

        self.play(FadeIn(large_block))
        self.play(large_block.animate.move_to(self.grid["F3"] + DOWN*0.05), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the first collision between blocks and the wall, starting a '1' counter (#FFFF00).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        counter_val = Integer(0, color="#FFFF00")
        counter_label = Text("Collisions:", font_size=24, color="#FFFF00")
        
        # Adjusting counter positioning to avoid clipping (Issue 30)
        self.place_at_grid(counter_label, "A4")
        self.place_at_grid(counter_val, "A5")
        
        self.play(Write(counter_label), Write(counter_val))
        
        # First impact: Large block hits Small block
        impact_point_1 = (self.grid["F2"] + self.grid["F3"]) / 2 + DOWN*0.05
        self.play(large_block.animate.shift(LEFT * 0.4), run_time=0.5)
        counter_val.set_value(1)
        self.play(Flash(impact_point_1, color="#FFFF00", flash_radius=0.5))
        
        # Second impact: Small block hits the wall
        impact_point_2 = np.array([self.grid["F1"][0] - 0.4, self.grid["F1"][1], 0])
        self.play(small_block.animate.shift(LEFT * 0.7), run_time=0.4)
        counter_val.set_value(2)
        self.play(Flash(impact_point_2, color="#FFFF00", flash_radius=0.5))
        
        self.wait(2)
