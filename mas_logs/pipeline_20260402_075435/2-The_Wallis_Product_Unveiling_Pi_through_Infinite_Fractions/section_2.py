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
        # Setup Title and Lecture Lines
        title_text = "Prerequisite: The Wallis Integrals"
        lecture_lines = [
            "We define the integral of sine to the nth power.",
            "The area under sine starts as a large wave.",
            "Increasing the exponent causes the area to shrink.",
            "Higher powers narrow into a thin spike.",
            "This specific area is denoted as I sub n."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display the integral I_n = integral from 0 to pi/2 of sin^n(x) dx in white (#FFFFFF)
        # Fix 37: wallis_formula in area A1 to B6
        # Using Text with simplified notation to ensure reliability without LaTeX
        wallis_formula = Text("I_n = ∫ sin^n(x) dx from 0 to π/2", font_size=28, color="#FFFFFF")
        self.place_in_area(wallis_formula, 'A1', 'B6', scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(wallis_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot y = sin(x) in orange (#FF7043) and y = sin^10(x) in blue (#2196F3) simultaneously.
        # Shade the area under y = sin(x) in orange (#FF7043)
        # Fix 38: sine_area_graph in area C1 to E6
        
        axes = Axes(
            x_range=[0, PI/2 + 0.3, PI/4],
            y_range=[0, 1.2, 0.5],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": GREY}
        )
        
        sin_1 = axes.plot(lambda x: np.sin(x), x_range=[0, PI/2], color="#FF7043")
        sin_10 = axes.plot(lambda x: np.sin(x)**10, x_range=[0, PI/2], color="#2196F3")
        area_1 = axes.get_area(sin_1, x_range=[0, PI/2], color="#FF7043", opacity=0.4)
        
        # VGroup for combined placement
        sine_area_graph = VGroup(axes, sin_1, sin_10, area_1)
        self.place_in_area(sine_area_graph, 'C1', 'E6', scale_factor=1.0)
        
        self.play(self.lecture[1].animate.set_color("#FF7043"))
        self.play(
            Create(axes),
            Create(sin_1),
            Create(sin_10),
            FadeIn(area_1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Increasing the exponent causes the area to shrink.
        self.play(self.lecture[2].animate.set_color("#2196F3"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Higher powers narrow into a thin spike.
        # Animate the transition of shading to the narrow spike of y = sin^10(x) in blue (#2196F3).
        area_10 = axes.get_area(sin_10, x_range=[0, PI/2], color="#2196F3", opacity=0.4)
        
        self.play(self.lecture[3].animate.set_color("#2196F3"))
        self.play(Transform(area_1, area_10))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This specific area is denoted as I sub n.
        # Label the current shaded area as 'I_n' and make it pulse (#FFFFFF).
        # Fix 39: notation at F4
        notation = Text("I_n", font_size=32, color="#FFFFFF")
        self.place_at_grid(notation, 'F4', scale_factor=1.0)
        
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(Write(notation))
        
        # Pulse animation
        self.play(
            notation.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=0.6
        )
        self.play(
            notation.animate.scale(1.3),
            rate_func=there_and_back,
            run_time=0.6
        )
        
        self.wait(3)
