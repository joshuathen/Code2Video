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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("How Facts are Coded in Weights", [
            "Facts are encoded in weight vectors.",
            "Inputs push weights toward specific outputs.",
            "This optimization happens during training."
        ])
        
        # Grid/Matrix visual representation
        matrix = VGroup()
        for i in range(16):
            rect = Square(side_length=0.4, fill_opacity=0.6, stroke_width=1)
            val = np.random.uniform(-1, 1)
            rect.set_fill(BLUE if val < 0 else RED)
            matrix.add(rect)
        matrix.arrange_in_grid(4, 4, buff=0.1)
        
        # Load asset
        neuron_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        self.place_at_grid(neuron_asset, "B5", scale_factor=0.5)
        neuron_asset.set_color(WHITE) # Initial color
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(matrix, "C2", "F4", scale_factor=0.8)
        self.play(Create(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight a sub-block of weights
        sub_block = VGroup(matrix[5], matrix[6], matrix[9], matrix[10])
        highlight = SurroundingRectangle(sub_block, color=YELLOW, buff=0.05)
        
        self.play(Create(highlight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pulse animation + Flash neuron
        self.play(
            sub_block.animate.scale(1.2).set_fill(opacity=0.9),
            neuron_asset.animate.set_color("#00FF00"),
            run_time=0.5
        )
        self.play(
            sub_block.animate.scale(1/1.2).set_fill(opacity=0.6),
            run_time=0.5
        )
        self.wait(2)
