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
        lecture_lines = [
            "Snell’s Law calculates the bending angle.",
            "It uses indices and sine values.",
            "The normal line is our reference.",
            "Compare the incident and refractive angles.",
            "Geometry predicts the path of light."
        ]
        self.setup_layout("Snell’s Law: The Mathematical Geometry", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw two media layers. Replaced missing SVG files with representative shapes.
        air_icon = Square(color="#F0F8FF", fill_opacity=0.5).scale(0.3)
        water_icon = Circle(color="#00BFFF", fill_opacity=0.5).scale(0.3)
        
        air = Rectangle(width=3, height=2, fill_opacity=0.3, fill_color="#F0F8FF", stroke_width=0)
        water = Rectangle(width=3, height=2, fill_opacity=0.3, fill_color="#00BFFF", stroke_width=0)
        
        self.place_in_area(air, "A4", "C6", scale_factor=0.8)
        self.place_in_area(water, "D4", "F6", scale_factor=0.8)
        
        self.play(FadeIn(air), FadeIn(water))
        self.lecture[0].set_color("#F0F8FF")

        # === Animation for Lecture Line 2 ===
        # Draw normal line
        normal = DashedLine(start=np.array([4.5, 2.5, 0]), end=np.array([4.5, -2.5, 0]), color="#808080")
        self.play(Create(normal))
        self.lecture[1].set_color("#808080")

        # === Animation for Lecture Line 3 ===
        # Incident ray
        incident_ray = Line(start=np.array([2.5, 1.5, 0]), end=np.array([4.5, 0, 0]), color="#FFD700", stroke_width=4)
        theta1_label = MathTex(r"\\theta_1", color="#FFD700").scale(0.8).next_to(incident_ray.get_end(), UP+LEFT, buff=0.1)
        self.play(Create(incident_ray), Write(theta1_label))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        # Refracted ray
        refracted_ray = Line(start=np.array([4.5, 0, 0]), end=np.array([6.0, -1.2, 0]), color="#32CD32", stroke_width=4)
        theta2_label = MathTex(r"\\theta_2", color="#32CD32")
        self.place_at_grid(theta2_label, "E5", scale_factor=0.9)
        self.play(Create(refracted_ray), Write(theta2_label))
        self.lecture[3].set_color("#32CD32")

        # === Animation for Lecture Line 5 ===
        # Equation
        equation = MathTex(r"n_1 \\sin(\\theta_1) = n_2 \\sin(\\theta_2)", color=WHITE)
        self.place_at_grid(equation, "A3", scale_factor=0.6)
        self.play(Write(equation))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
