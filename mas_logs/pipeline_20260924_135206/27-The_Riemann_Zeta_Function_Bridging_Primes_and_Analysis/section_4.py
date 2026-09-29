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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Critical Strip and the Hypothesis", 
                          ["The critical strip lies between zero and one.", 
                           "Riemann proposed all zeros lie on line half.", 
                           "Zeros act like nodes on a guitar string."])
        
        # --- Assets ---
        guitar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg")

        # --- Visualization Elements ---
        # Critical Strip
        strip = Rectangle(width=1.6, height=4.8, color="#D3D3D3", fill_opacity=0.3)
        self.place_in_area(strip, "B3", "E5", scale_factor=0.9)
        
        # Center Line (Re(s)=1/2)
        ref_line = Line(start=self.grid["B4"], end=self.grid["E4"], color="#FFFFFF", stroke_width=4)
        
        # Zeros (nodes)
        node_1 = Dot(color="#FF0000", radius=0.1)
        node_2 = Dot(color="#FF0000", radius=0.1)
        self.place_at_grid(node_1, "C4", scale_factor=0.5)
        self.place_at_grid(node_2, "D4", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # Draw the critical strip and guitar icon
        self.play(FadeIn(strip), FadeIn(self.place_at_grid(guitar, "A6", scale_factor=0.3)))
        self.lecture[0].set_color("#D3D3D3")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the line Re(s)=1/2
        self.play(Create(ref_line))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Zeros act like nodes
        self.play(FadeIn(node_1), FadeIn(node_2))
        self.play(
            node_1.animate.shift(UP * 0.2), 
            node_2.animate.shift(DOWN * 0.2), 
            run_time=1, rate_func=there_and_back
        )
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
