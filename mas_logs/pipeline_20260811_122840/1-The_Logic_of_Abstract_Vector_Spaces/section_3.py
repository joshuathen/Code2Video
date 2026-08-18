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
        lecture_lines = ["Polynomials form a vector space.", "Adding polynomials adds their DNA.", "Scaling stretches the polynomial curve."]
        self.setup_layout("Example: The Polynomial Vector Space", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[-2, 2], y_range=[-2, 4], axis_config={"include_tip": False}).scale(0.5)
        p_curve = axes.plot(lambda x: 0.5*x**2 + x + 1, color="#FF4500")
        p_label = MathTex("p(x) = ax^2 + bx + c", color="#FF4500").scale(0.7)
        dna_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg")
        
        self.place_in_area(VGroup(axes, p_curve), "A4", "C6", scale_factor=0.8)
        self.place_in_area(p_label, "B2", "C3", scale_factor=0.8)
        self.place_at_grid(dna_icon, "D2", scale_factor=0.6)
        dna_icon.next_to(p_label, RIGHT, buff=0.2)
        
        self.play(Create(axes), Create(p_curve), Write(p_label), FadeIn(dna_icon))
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        dna_markers = VGroup(
            Vector([0, 1, 0], color="#32CD32"),
            Vector([1, 0, 0], color="#32CD32"),
            Vector([0, 0, 1], color="#32CD32")
        ).arrange(RIGHT)
        
        dna_label = Text("DNA Markers: (a, b, c)", font_size=20, color="#32CD32")
        
        self.place_at_grid(dna_markers, "E3", scale_factor=0.8)
        self.place_at_grid(dna_label, "D4", scale_factor=0.7)
        
        self.play(FadeIn(dna_markers), Write(dna_label))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        axes3d = ThreeDAxes(x_range=[-1, 2], y_range=[-1, 2], z_range=[-1, 2]).scale(0.3)
        vectors = VGroup(*[
            Arrow3D(start=ORIGIN, end=np.array([1.0, 0.0, 0.0]), color="#FFFF00"),
            Arrow3D(start=ORIGIN, end=np.array([0.0, 1.0, 0.0]), color="#FFFF00"),
            Arrow3D(start=ORIGIN, end=np.array([0.0, 0.0, 1.0]), color="#FFFF00")
        ])
        dna_icon_3d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg").scale(0.5)
        
        self.place_in_area(axes3d, "B4", "E6", scale_factor=0.9)
        self.place_at_grid(vectors, "C5", scale_factor=0.8)
        self.place_at_grid(dna_icon_3d, "E5", scale_factor=0.8)
        
        self.play(FadeIn(axes3d), Create(vectors), FadeIn(dna_icon_3d))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
