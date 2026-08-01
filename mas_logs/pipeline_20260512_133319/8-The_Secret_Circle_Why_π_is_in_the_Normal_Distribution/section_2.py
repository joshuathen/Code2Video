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
        # Data
        title = "Prerequisite: The Gaussian Function"
        lines = [
            "Let's examine the core of this Gaussian function.",
            "We need the total area under this infinite curve.",
            "Unfortunately, no simple antiderivative exists in one dimension."
        ]
        
        # Colors
        SALMON = "#FC6255"
        RED_X = "#FF0000"
        WHITE_C = "#FFFFFF"

        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Formula: f(x) = e⁻ˣ²
        self.lecture[0].set_color(WHITE_C)
        formula = Text("f(x) = e⁻ˣ²", font_size=40, color=WHITE_C)
        self.place_in_area(formula, "A2", "A5")
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Curve and shaded area
        self.lecture[1].set_color(SALMON)
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": True, "font_size": 18},
            tips=False
        )
        # Resolve Issue 30: Adjusting area from B1-E6 to B2-E6 to avoid lecture notes obstruction
        self.place_in_area(axes, "B2", "E6", scale_factor=0.6)
        
        curve = axes.plot(lambda x: np.exp(-x**2), color=SALMON, x_range=[-3, 3])
        area = axes.get_area(curve, x_range=[-3, 3], color=SALMON, opacity=0.3)
        
        self.play(Create(axes))
        self.play(Create(curve), run_time=2)
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Integral and No antiderivative label with red X
        self.lecture[2].set_color(RED_X)
        
        integral_label = Text("I = ∫ e⁻ˣ² dx (from -∞ to ∞)", font_size=24, color=WHITE_C)
        # Resolve Issue 31: Adjusted area to F1-F4 and scale to 0.8 to prevent overflow
        self.place_in_area(integral_label, "F1", "F4", scale_factor=0.8)
        
        no_antideriv_text = Text("No elementary antiderivative", font_size=20, color=RED_X)
        # Resolve Issue 32: Adjusted area to F5-F6 and scale to 0.7 to prevent overlap and edge clipping
        self.place_in_area(no_antideriv_text, "F5", "F6", scale_factor=0.7)
        
        # Red X over the label
        cross_line1 = Line(start=no_antideriv_text.get_corner(UL), end=no_antideriv_text.get_corner(DR), color=RED_X, stroke_width=4)
        cross_line2 = Line(start=no_antideriv_text.get_corner(UR), end=no_antideriv_text.get_corner(DL), color=RED_X, stroke_width=4)
        cross_x = VGroup(cross_line1, cross_line2)

        self.play(Write(integral_label))
        self.play(FadeIn(no_antideriv_text))
        self.play(Create(cross_x))
        self.wait(2)
