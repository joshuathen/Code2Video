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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize lecture lines
        lecture_lines_text = [
            "In explicit functions, y stands alone, like y = 3x+2.",
            "Implicit equations mix x and y together, like a knot.",
            "Consider a circle's equation: x^2 + y^2 = 25.",
            "Finding the slope here isn't as straightforward as before.",
            "How do we differentiate without isolating y first?"
        ]
        
        self.setup_layout("The Mystery of the Tangled Equation", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # Show titles 'Explicit' (#00FF00) and equation 'y = 3x + 2' (#00FF00)
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        explicit_title = Text("Explicit", color="#00FF00", font_size=24)
        self.place_in_area(explicit_title, "A1", "A3")
        
        explicit_eq = Text("y = 3x + 2", color="#00FF00")
        self.place_in_area(explicit_eq, "B1", "B3")
        
        self.play(FadeIn(explicit_title), Write(explicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show title 'Implicit' (#FF00FF) and 'x^2 + y^2 = 25' (#FF00FF).
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        implicit_title = Text("Implicit", color="#FF00FF", font_size=24)
        self.place_in_area(implicit_title, "A4", "A6")
        
        # Using a single Text object for environment compatibility (MathTex requires local LaTeX install)
        implicit_eq = Text("x^2 + y^2 = 25", color="#FF00FF")
        self.place_in_area(implicit_eq, "B4", "B6")
        
        self.play(FadeIn(implicit_title), Write(implicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Plot the graph of y = sqrt(25 - x^2) as a green semi-circle (#00FF00).
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Define axes area on the right side
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            axis_config={"include_tip": False},
            x_length=4.5,
            y_length=3.0
        )
        self.place_in_area(axes, "C1", "F6")
        
        # Semi-circle plot (green)
        semi_circle = axes.plot(
            lambda x: np.sqrt(np.maximum(0, 25 - x**2)), 
            x_range=[-5, 5], 
            color="#00FF00"
        )
        
        self.play(Create(axes), Create(semi_circle))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transform the semi-circle into a full yellow circle (#FFFF00).
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Full circle plot (yellow) using parametric curve
        full_circle = axes.plot_parametric_curve(
            lambda t: np.array([5 * np.cos(t), 5 * np.sin(t), 0]),
            t_range=[0, TAU],
            color="#FFFF00"
        )
        
        self.play(Transform(semi_circle, full_circle))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the 'y^2' term in the implicit equation in red (#FF0000).
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        
        # Flash the y^2 part (index roughly targets the 7th/8th characters 'y^2' in the Text object)
        self.play(Indicate(implicit_eq[4:6], color="#FF0000", scale_factor=1.5))
        self.wait(2)