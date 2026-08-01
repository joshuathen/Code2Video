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
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content - Using Text instead of Tex to avoid LaTeX dependency issues
        lecture_texts = [Text(line, font_size=18, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.scale(0.85).to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # 6x6 grid on the right side for visualization
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset to the right half of the screen
                x = 1.0 + j * 0.9
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # Set up the lecture frame
        # Strings modified to remove LaTeX syntax and use Unicode where appropriate
        self.setup_layout(
            "Snell's Law via Fermat's Principle",
            [
                "- Fermat: Light takes path of least time",
                "- Total Time T = L1/v1 + L2/v2",
                "- v = c/n => T = 1/c (n1 L1 + n2 L2)",
                "- Minimize T(x) by setting dT/dx = 0",
                "- Result: n1 sin θ1 = n2 sin θ2"
            ]
        )

        # Optical Interface
        interface = Line(LEFT * 2.5, RIGHT * 2.5, color=BLUE)
        self.place_at_grid(interface, "C3")
        interface.shift(RIGHT * 0.5)

        # Normal Line
        normal = DashedLine(UP * 2, DOWN * 2, color=GRAY)
        self.place_at_grid(normal, "C3")
        normal.shift(RIGHT * 0.5)

        # Calculation of hit point based on grid
        hit_point = self.grid["C3"] + RIGHT * 0.5
        
        # Rays
        start_point = hit_point + LEFT * 1.5 + UP * 1.8
        end_point = hit_point + RIGHT * 1.2 + DOWN * 1.8
        
        incident_ray = Line(start_point, hit_point, color=YELLOW)
        refracted_ray = Line(hit_point, end_point, color=YELLOW)

        # Labels (Converted from MathTex to Text)
        n1_label = Text("n1", color=WHITE, font_size=24)
        self.place_at_grid(n1_label, "B5", 0.8)
        
        n2_label = Text("n2", color=WHITE, font_size=24)
        self.place_at_grid(n2_label, "E5", 0.8)

        # Angles (Converted from MathTex to Text)
        theta1 = Text("θ1", font_size=20).move_to(hit_point + UP * 0.7 + LEFT * 0.25)
        theta2 = Text("θ2", font_size=20).move_to(hit_point + DOWN * 0.7 + RIGHT * 0.25)

        # Animations
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=UP))
        self.play(Create(interface), Create(normal))
        self.wait(0.5)
        
        self.play(Create(incident_ray))
        self.play(Create(refracted_ray))
        self.play(
            Write(n1_label), 
            Write(n2_label),
            Write(theta1),
            Write(theta2)
        )
        
        # Final formula highlight (Converted from MathTex to Text)
        final_formula = Text("n1 sin θ1 = n2 sin θ2", color=YELLOW, font_size=32)
        final_formula.next_to(self.title, DOWN, buff=0.3)
        self.play(Write(final_formula))
        
        self.wait(3)
