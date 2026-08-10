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
        lecture_lines = [
            "Superposition combines basis states linearly.",
            "The state is a sum: |ψ> equals α|0> plus β|1>.",
            "Visualize this arrow pointing anywhere on a sphere.",
            "Like a chord blending notes C, E, and G.",
            "The state lives on the Bloch sphere surface."
        ]
        self.setup_layout("Defining Superposition: The Linear Combination", lecture_lines)
        
        # Assets
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        piano_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/piano.svg")
        
        # Define objects
        basis_0 = Arrow(ORIGIN, UP, color="#D3D3D3", buff=0)
        basis_1 = Arrow(ORIGIN, RIGHT, color="#D3D3D3", buff=0)
        psi = Arrow(ORIGIN, UP*0.7 + RIGHT*0.7, color=YELLOW, buff=0)
        
        # Group together as per B002
        basis_group = VGroup(basis_0, basis_1, psi, sphere_icon)
        self.place_at_grid(basis_group, 'C5', scale_factor=0.6)
        
        # Add labels
        label_0 = Text("|0>", color="#D3D3D3", font_size=20).next_to(basis_0, UP)
        label_1 = Text("|1>", color="#D3D3D3", font_size=20).next_to(basis_1, RIGHT)
        label_psi = Text("|ψ>", color=YELLOW, font_size=20).next_to(psi.get_end(), UR, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), FadeIn(sphere_icon), FadeIn(basis_group), FadeIn(label_0), FadeIn(label_1))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), FadeIn(psi), FadeIn(label_psi))
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(Rotate(psi, angle=PI/4, about_point=ORIGIN))
        self.play(self.lecture[2].animate.set_color(BLUE))

        # === Animation for Lecture Line 4 ===
        piano_icon.scale(0.5).next_to(self.lecture[3], RIGHT)
        self.play(FadeIn(self.lecture[3]), FadeIn(piano_icon))
        self.play(self.lecture[3].animate.set_color(BLUE))

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.play(Indicate(psi))
        self.play(self.lecture[4].animate.set_color(BLUE))
        
        self.wait(2)
