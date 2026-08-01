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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "Setting the Rules: Elastic Collisions", 
            [
                "We assume no friction and perfectly elastic collisions.",
                "Momentum and kinetic energy are fully conserved.",
                "The wall is immovable, reflecting the small block."
            ]
        )
        
        # Define colors
        COLOR_BLOCK_1 = "#1890FF" # Blue
        COLOR_BLOCK_2 = "#52C41A" # Green
        COLOR_WALL = "#FFFFFF"    # White
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow

        # Load Assets (Issue 22)
        BLOCK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        
        # Initialize blocks using SVG (Issue 22)
        # Block 1 (Small) at D3 (Issue 39 fix: moved from D2 to D3)
        block_1 = SVGMobject(BLOCK_ASSET, color=COLOR_BLOCK_1, fill_opacity=0.8)
        self.place_at_grid(block_1, "D3", scale_factor=0.4)
        
        # Block 2 (Large) at D5 (Issue 41 fix: moved from D4 to D5)
        block_2 = SVGMobject(BLOCK_ASSET, color=COLOR_BLOCK_2, fill_opacity=0.8)
        self.place_at_grid(block_2, "D5", scale_factor=0.8)

        # Initialize Wall
        # Wall is a vertical line on the far left side of the right area (Col 1)
        wall_top = self.grid["A1"] + UP * 0.5
        wall_bottom = self.grid["F1"] + DOWN * 0.5
        wall = Line(wall_top, wall_bottom, color=COLOR_WALL, stroke_width=6)
        
        # === Animation for Lecture Line 1 ===
        # "We assume no friction and perfectly elastic collisions."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(block_1), Create(block_2))
        self.play(Flash(block_1, color=WHITE), Flash(block_2, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Momentum and kinetic energy are fully conserved."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Velocity vectors (arrows)
        # Vector for block 1 pointing left
        arrow_1 = Arrow(
            start=block_1.get_center(), 
            end=block_1.get_center() + LEFT * 1.5, 
            color=COLOR_BLOCK_1, 
            buff=0
        )
        # Vector for block 2 pointing left
        arrow_2 = Arrow(
            start=block_2.get_center(), 
            end=block_2.get_center() + LEFT * 2, 
            color=COLOR_BLOCK_2, 
            buff=0
        )
        
        # Labels for conservation
        # Fixes from Issue 40:
        # Move p_label to C3 and ke_label to C5
        p_label = MathTex("p = mv", font_size=24, color=WHITE)
        ke_label = MathTex("KE = \\frac{1}{2}mv^2", font_size=24, color=WHITE)
        self.place_at_grid(p_label, 'C3', scale_factor=0.8)
        self.place_at_grid(ke_label, 'C5', scale_factor=0.8)

        self.play(Create(arrow_1), Create(arrow_2), Write(p_label), Write(ke_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The wall is immovable, reflecting the small block."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        self.play(Create(wall))
        self.play(wall.animate.set_stroke(width=10), run_time=0.5)
        self.play(wall.animate.set_stroke(width=6), run_time=0.5)
        
        # Reflect block 1 vector to show wall interaction concept
        arrow_1_reflected = Arrow(
            start=block_1.get_center(), 
            end=block_1.get_center() + RIGHT * 1.5, 
            color=COLOR_BLOCK_1, 
            buff=0
        )
        
        self.play(Transform(arrow_1, arrow_1_reflected))
        self.wait(2)

        # Final cleanup for the section: reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
