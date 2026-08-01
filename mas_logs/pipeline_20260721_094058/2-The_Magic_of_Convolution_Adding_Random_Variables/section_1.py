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
        # Title and Lecture Lines
        title_text = "Prerequisite: The Independent Duo"
        lecture_lines = [
            "Start with two independent random variables, X and Y.",
            "We want to find the sum of their distributions.",
            "Let's visualize this relationship on a 2D grid."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show two icons representing Alice (X) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg] 
        # and Bob (Y) in #ADD8E6 and #FFB6C1.
        line1_color = "#ADD8E6"
        alice_color = "#ADD8E6"
        bob_color = "#FFB6C1"
        
        self.play(self.lecture[0].animate.set_color(line1_color))
        
        # Alice Icon using provided asset
        alice_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg").set_color(alice_color)
        alice_label = MathTex("X", color=WHITE).scale(1.2)
        alice_icon = VGroup(alice_svg, alice_label)
        # Fix for Issue 19: Scale factor 0.9 at B2
        self.place_at_grid(alice_icon, "B2", scale_factor=0.9)
        
        # Bob Icon (No asset provided for Bob, using circle as placeholder icon)
        bob_circle = Circle(radius=0.5, color=bob_color, fill_opacity=0.6)
        bob_label = MathTex("Y", color=WHITE).scale(1.2)
        bob_icon = VGroup(bob_circle, bob_label)
        # Fix for Issue 19: Scale factor 0.9 at B5
        self.place_at_grid(bob_icon, "B5", scale_factor=0.9)
        
        alice_name = Text("Alice", font_size=20, color=alice_color)
        bob_name = Text("Bob", font_size=20, color=bob_color)
        # Position names within 1 grid unit
        alice_name.next_to(alice_icon, DOWN, buff=0.1)
        bob_name.next_to(bob_icon, DOWN, buff=0.1)
        
        self.play(FadeIn(alice_icon), FadeIn(alice_name))
        self.play(FadeIn(bob_icon), FadeIn(bob_name))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade in the equation 'Z = X + Y' in #FFFFFF at the top center.
        line2_color = "#FFFFFF"
        self.play(self.lecture[1].animate.set_color(line2_color))
        
        equation = MathTex("Z = X + Y", font_size=42, color=line2_color)
        # Fix for Issue 18: Position at B3-B4 instead of A3-A4
        self.place_in_area(equation, "B3", "B4", scale_factor=1.0)
        
        self.play(FadeIn(equation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a 2D coordinate grid with X and Y axes; highlight cells in #FFFFE0.
        line3_color = "#FFFFE0"
        self.play(self.lecture[2].animate.set_color(line3_color))
        
        # Grid construction (simple 2D grid)
        grid_size = 4
        square_side = 0.6
        grid_squares = VGroup(*[
            Square(side_length=square_side, stroke_color=WHITE, stroke_width=2)
            for _ in range(grid_size * grid_size)
        ]).arrange_in_grid(rows=grid_size, cols=grid_size, buff=0)
        
        # Axis labels
        x_axis_label = MathTex("X", color=alice_color).scale(0.8)
        y_axis_label = MathTex("Y", color=bob_color).scale(0.8)
        
        # Fix for Issue 17: Position at C2 to E5 instead of C2 to F5
        self.place_in_area(grid_squares, "C2", "E5", scale_factor=1.0)
        
        x_axis_label.next_to(grid_squares, DOWN, buff=0.2)
        y_axis_label.next_to(grid_squares, LEFT, buff=0.2)
        
        self.play(Create(grid_squares), Write(x_axis_label), Write(y_axis_label))
        
        # Highlight some cells (e.g., center 4)
        highlights = VGroup(*[
            grid_squares[i].copy().set_fill(line3_color, opacity=0.4).set_stroke(width=0)
            for i in [5, 6, 9, 10]
        ])
        
        self.play(FadeIn(highlights))
        self.wait(2)
