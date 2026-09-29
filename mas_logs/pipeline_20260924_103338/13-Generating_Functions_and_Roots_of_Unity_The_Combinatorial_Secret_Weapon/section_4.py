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
        self.setup_layout("Summary and Visualization Strategy", ["Encode sequences into polynomials.", "Rotate using roots of unity.", "Average results to isolate specific sums."])
        
        # === Animation for Lecture Line 1 ===
        # Encode sequences into polynomials.
        summary_table = VGroup(
            Text("Algebraic: P(x) = \u2211 a_n x^n", font_size=20),
            Text("Geometric: Vector sum in C", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(summary_table, 'B4', scale_factor=0.6)
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(summary_table))

        # === Animation for Lecture Line 2 ===
        # Rotate using roots of unity.
        root_circle = Circle(radius=1.0, color="#FF00FF")
        vector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        vector.set_color("#FFCC00")
        
        roots = VGroup(*[Dot(point=np.array([np.cos(2*np.pi*k/5), np.sin(2*np.pi*k/5), 0])) for k in range(5)])
        rotation_viz = VGroup(root_circle, roots, vector)
        self.place_at_grid(rotation_viz, 'D5', scale_factor=0.5)
        self.lecture[1].set_color("#FFCC00")
        
        self.play(FadeIn(rotation_viz), Create(root_circle))
        self.play(Rotate(vector, angle=2*PI, about_point=self.grid['D5']))

        # === Animation for Lecture Line 3 ===
        # Average results to isolate specific sums.
        final_diagram = VGroup(
            Line(LEFT, RIGHT),
            Text("Sum / N", font_size=20)
        ).arrange(DOWN)
        self.place_at_grid(final_diagram, 'F3', scale_factor=0.7)
        self.lecture[2].set_color("#FF00CC")
        self.play(FadeIn(final_diagram))
        self.wait(2)
