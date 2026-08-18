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
        self.setup_layout("Defining the Transition Matrix", [
            "Transition matrix P maps bases.", 
            "Columns of P are new basis vectors.", 
            "[v]_B = P * [v]_B' relates coordinates."
        ])
        self.lecture.set_opacity(1)

        # === Animation for Lecture Line 1 ===
        # Draw vectors v_A and v_B in #FFFFFF
        vA = Vector([1, 1.5], color=WHITE)
        vB = Vector([1.5, 0.5], color=WHITE)
        self.place_at_grid(vA, 'D2', scale_factor=0.8)
        self.place_at_grid(vB, 'E3', scale_factor=0.8)
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(vA), Create(vB))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight basis vectors of system A in #FFFF00
        basisA1 = Vector([1, 0], color=YELLOW)
        basisA2 = Vector([0, 1], color=YELLOW)
        self.place_at_grid(basisA1, 'D1', scale_factor=0.7)
        self.place_at_grid(basisA2, 'A2', scale_factor=0.8)
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(basisA1), Create(basisA2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show projection and bridge as asset
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge, 'B5', scale_factor=0.8)
        self.lecture[2].set_color("#00FF00")
        self.play(FadeIn(bridge))
        
        # Represent v in B-coordinates entering bridge and outputting v in A
        vector_v_in = Arrow(start=self.grid['F5'], end=self.grid['D5'], color=BLUE)
        self.play(Create(vector_v_in))
        self.play(Transform(vector_v_in, Arrow(start=self.grid['B5'], end=self.grid['A5'], color=GREEN)))
        
        self.wait(2)
