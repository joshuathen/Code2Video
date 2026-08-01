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
        # Setup the layout with specific title and lecture lines as defined in the section metadata
        self.setup_layout("The Mystery of the Tangled Equation", [
            "Explicit functions keep y isolated on one side.",
            "Implicit equations mix x and y together.",
            "Take this circle: solving for y creates messy roots.",
            "Yet, every point on this curve has a slope.",
            "Implicit differentiation reveals slope without isolating y."
        ])

        # Color palette
        BLUE_EXPLICIT = "#58C4DD"
        YELLOW_IMPLICIT = "#FFFF00"
        RED_X = "#FF0000"
        WHITE_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line
        self.play(self.lecture[0].animate.set_color(BLUE_EXPLICIT))
        
        # Create expressions 'y = 2x + 1' (Blue) and 'x² + y² = 25' (Yellow)
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        explicit_eq = Text("y = 2x + 1", color=BLUE_EXPLICIT)
        implicit_eq = Text("x² + y² = 25", color=YELLOW_IMPLICIT)
        
        self.place_at_grid(explicit_eq, "A2", scale_factor=0.8)
        self.place_at_grid(implicit_eq, "A5", scale_factor=0.8)
        
        self.play(FadeIn(explicit_eq), FadeIn(implicit_eq))
        
        # Pulsing white circle around 'y' in the explicit equation
        # Indexing into Text (VGroup of chars) to target the 'y' character
        y_char = explicit_eq[0]
        pulse_circle = Circle(radius=0.25, color=WHITE).move_to(y_char.get_center())
        
        self.play(Create(pulse_circle))
        self.play(pulse_circle.animate.scale(1.4), rate_func=there_and_back)
        self.play(FadeOut(pulse_circle))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight the second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE_TEXT),
            self.lecture[1].animate.set_color(YELLOW_IMPLICIT)
        )
        
        # Emphasize the "mixed" nature of the implicit equation
        self.play(implicit_eq.animate.scale(1.2), rate_func=there_and_back)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight the third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE_TEXT),
            self.lecture[2].animate.set_color(YELLOW_IMPLICIT)
        )
        
        # Draw a large yellow circle centered in the right-side work area
        circle_obj = Circle(radius=1.8, color=YELLOW_IMPLICIT)
        self.place_in_area(circle_obj, "B1", "F6")
        
        # Show messy root text appearing using Text and Unicode symbols
        messy_root = Text("y = ± √(25 - x²)", color=WHITE_TEXT)
        self.place_at_grid(messy_root, "C4", scale_factor=0.9)
        
        # Cover with a red 'X' to indicate we want to avoid this approach
        red_x = Cross(messy_root, stroke_color=RED_X)
        
        self.play(Create(circle_obj))
        self.play(Write(messy_root))
        self.play(Create(red_x))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Transition highlights
        self.play(
            self.lecture[2].animate.set_color(WHITE_TEXT),
            self.lecture[3].animate.set_color(WHITE_TEXT)
        )
        
        # Show dots on the circle to indicate points where slope exists
        indicator_points = [circle_obj.point_at_angle(a * DEGREES) for a in [45, 135, 225, 315]]
        dots = VGroup(*[Dot(p, color=WHITE_TEXT, radius=0.08) for p in indicator_points])
        
        self.play(FadeIn(dots))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Highlight final line of the section
        self.play(
            self.lecture[3].animate.set_color(WHITE_TEXT),
            self.lecture[4].animate.set_color(WHITE_TEXT)
        )
        
        # Display a large white question mark at the center of the circle curve
        question_mark = Text("?", font_size=100, color=WHITE_TEXT)
        question_mark.move_to(circle_obj.get_center())
        
        # Remove intermediate clutter and reveal the question mark
        self.play(
            FadeOut(messy_root),
            FadeOut(red_x),
            FadeOut(dots)
        )
        self.play(Write(question_mark))
        self.wait(2)
