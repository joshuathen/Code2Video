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
        title = "The Foundation: Understanding Independence"
        lines = [
            "Events are independent if they don't affect each other.",
            "A coin flip doesn't change a die roll result.",
            "One outcome gives no information about the other.",
            "Their joint probability is the product of individual probabilities.",
            "Visually, the intersection area matches this product."
        ]
        self.setup_layout(title, lines)

        # Colors
        COIN_COLOR = "#ADD8E6"
        DIE_COLOR = "#FFB6C1"
        FORMULA_COLOR = "#FFFFFF"
        NO_INFO_COLOR = "#FF0000"
        INTERSECTION_COLOR = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Display a coin icon (Circle labeled A, #ADD8E6) on the left and a die icon (Square labeled B, #FFB6C1) on the right.
        self.lecture[0].set_color(COIN_COLOR)
        
        coin_circle = Circle(radius=0.4, color=COIN_COLOR, fill_opacity=0.3)
        coin_label = Text("A", font_size=24, color=COIN_COLOR)
        coin_group = VGroup(coin_circle, coin_label)
        # Fix for Issue 31: Line 70: self.place_at_grid(coin_group, 'C3', scale_factor=1.2)
        self.place_at_grid(coin_group, "C3", scale_factor=1.2)
        
        die_square = Square(side_length=0.8, color=DIE_COLOR, fill_opacity=0.3)
        die_label = Text("B", font_size=24, color=DIE_COLOR)
        die_group = VGroup(die_square, die_label)
        # Fix for Issue 32: Line 75: self.place_at_grid(die_group, 'C5', scale_factor=1.2)
        self.place_at_grid(die_group, "C5", scale_factor=1.2)
        
        self.play(Create(coin_group), Create(die_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the coin (A) flipping and the die (B) rolling simultaneously to show they act independently.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(DIE_COLOR)
        
        self.play(
            Rotate(coin_group, axis=RIGHT, angle=2*PI),
            Rotate(die_group, angle=2*PI),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a question mark (#FFFFFF) between A and B, then draw a red "X" (#FF0000) over it to signify no information flow.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(FORMULA_COLOR)
        
        q_mark = Text("?", font_size=48, color=WHITE)
        self.place_at_grid(q_mark, "C4")
        
        cross_line1 = Line(start=q_mark.get_corner(UL), end=q_mark.get_corner(DR), color=NO_INFO_COLOR, stroke_width=8)
        cross_line2 = Line(start=q_mark.get_corner(DL), end=q_mark.get_corner(UR), color=NO_INFO_COLOR, stroke_width=8)
        cross = VGroup(cross_line1, cross_line2)
        
        self.play(FadeIn(q_mark))
        self.play(Create(cross))
        self.wait(1)
        self.play(FadeOut(q_mark), FadeOut(cross))

        # === Animation for Lecture Line 4 ===
        # Display the formula "P(A \cap B) = P(A) \times P(B)" in the center of the screen in #FFFFFF.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(FORMULA_COLOR)
        
        formula = MathTex(r"P(A \cap B) = P(A) \times P(B)", color=FORMULA_COLOR, font_size=36)
        # Fix for Issue 33: Line 107: self.place_in_area(formula, 'E2', 'E5', scale_factor=1.2)
        self.place_in_area(formula, "E2", "E5", scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Draw two overlapping circles (A and B) where the intersection area pulses in #00FF00 to represent the product.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(INTERSECTION_COLOR)
        
        # Transform Square B to Circle B for Venn diagram
        die_circle = Circle(radius=0.4, color=DIE_COLOR, fill_opacity=0.3)
        die_circle.scale(1.2) # Match scale of coin_circle
        die_circle.move_to(die_group.get_center())
        
        self.play(
            ReplacementTransform(die_square, die_circle),
        )
        new_die_group = VGroup(die_circle, die_label)
        
        # Move to overlap at C4
        center_pos = self.grid["C4"]
        self.play(
            coin_group.animate.move_to(center_pos + LEFT * 0.3),
            new_die_group.animate.move_to(center_pos + RIGHT * 0.3),
            run_time=1.5
        )
        
        # Create intersection highlight
        intersect_area = Intersection(
            coin_circle, die_circle, 
            color=INTERSECTION_COLOR, fill_opacity=0.8
        )
        
        self.play(FadeIn(intersect_area))
        # Pulsing effect
        for _ in range(2):
            self.play(Indicate(intersect_area, color=INTERSECTION_COLOR, scale_factor=1.2))
        
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)
