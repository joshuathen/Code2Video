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
        # Title and Lecture Lines
        title = "The Ten Commandments (Axioms) Simplified"
        lecture_lines = [
            "Closure means adding two vectors stays in the set.",
            "Scalar multiplication must also keep results within the set.",
            "Every space must contain a unique zero vector.",
            "Adding the zero vector leaves other vectors unchanged.",
            "Rules like commutativity ensure the order doesn't matter."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        VSPACE_COLOR = "#ADD8E6"
        DOT_COLOR = "#FFFFFF"
        SUM_COLOR = "#FFFF00"
        ZERO_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Closure means adding two vectors stays in the set.
        self.lecture[0].set_color(VSPACE_COLOR)
        
        # Create a large circle representing the 'Vector Space'.
        space_circle = Circle(radius=2.4, color=VSPACE_COLOR, fill_opacity=0.1)
        self.place_in_area(space_circle, "A1", "F6")
        
        # Place text 'The Rules' at the top of the circle.
        rules_text = Text("The Rules", font_size=24, color=VSPACE_COLOR)
        self.place_in_area(rules_text, "A3", "A4", scale_factor=0.8)
        
        self.play(Create(space_circle), FadeIn(rules_text))

        # Animate two dots merging into a new dot within the boundary.
        dot1 = Dot(color=DOT_COLOR)
        dot2 = Dot(color=DOT_COLOR)
        self.place_at_grid(dot1, "B2")
        self.place_at_grid(dot2, "C4")
        
        dot3 = Dot(color=SUM_COLOR)
        self.place_at_grid(dot3, "B4")
        
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.wait(0.5)
        self.play(ReplacementTransform(VGroup(dot1, dot2), dot3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Scalar multiplication must also keep results within the set.
        self.lecture[1].set_color(SUM_COLOR)
        
        # Select a dot and scale its size.
        scalar_dot = Dot(color=DOT_COLOR)
        self.place_at_grid(scalar_dot, "E2")
        self.play(FadeIn(scalar_dot))
        
        # Scale and move - remains inside the circle boundary.
        self.play(scalar_dot.animate.scale(3))
        self.play(scalar_dot.animate.shift(RIGHT * 0.5 + DOWN * 0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every space must contain a unique zero vector.
        self.lecture[2].set_color(ZERO_COLOR)
        
        # Fade in a bright red point at the center of the circle.
        zero_dot = Dot(color=ZERO_COLOR)
        self.place_in_area(zero_dot, "C3", "D4") 
        
        zero_label = Text("Zero Vector: The Identity", font_size=18, color=ZERO_COLOR)
        self.place_at_grid(zero_label, "E4")
        
        self.play(FadeIn(zero_dot), Write(zero_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Adding the zero vector leaves other vectors unchanged.
        self.lecture[3].set_color(WHITE)
        
        # Visual demonstration: a white dot moves to the zero vector and returns.
        test_dot = Dot(color=DOT_COLOR)
        self.place_at_grid(test_dot, "B5")
        self.play(FadeIn(test_dot))
        
        # Move to zero and back
        self.play(test_dot.animate.move_to(zero_dot.get_center()), run_time=1)
        self.play(test_dot.animate.move_to(self.grid["B5"]), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Rules like commutativity ensure the order doesn't matter.
        self.lecture[4].set_color(VSPACE_COLOR)
        
        # Flash the entire circle's border to signify the 'Bubble' system.
        self.play(Flash(space_circle, color=WHITE, line_length=0.5, flash_radius=2.5))
        self.play(space_circle.animate.set_stroke(WHITE, width=6), run_time=0.4)
        self.play(space_circle.animate.set_stroke(VSPACE_COLOR, width=2), run_time=0.4)
        
        self.wait(2)
