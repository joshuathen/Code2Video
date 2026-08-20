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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Our geometric intuition fails in high dimensions.",
            "Mathematics reveals structures we cannot visualize.",
            "High-dimensional spheres drive modern AI logic."
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Summarize core concepts. Color: #FF4500.
        concept = Text("Intuition < Math", font_size=36, color="#FF4500")
        self.place_in_area(concept, "B1", "B6", scale_factor=0.9)
        self.play(FadeIn(concept))
        self.lecture[0].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Hidden math structures. Color: #00FFFF.
        formula = MathTex(r"V_n(R) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} R^n", color="#00FFFF")
        self.place_in_area(formula, "C2", "D5", scale_factor=0.85)
        self.play(FadeIn(formula))
        self.lecture[1].set_color("#00FFFF")
        self.wait(2)
        self.play(FadeOut(formula), FadeOut(concept))

        # === Animation for Lecture Line 3 ===
        # N-Spheres in AI. Color: #FFD700.
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FFD700")
        self.place_at_grid(sphere_icon, "A4", scale_factor=0.7)
        self.play(FadeIn(sphere_icon))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
        self.play(FadeOut(sphere_icon), FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)
