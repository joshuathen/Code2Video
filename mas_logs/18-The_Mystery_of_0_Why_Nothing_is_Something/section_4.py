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
        # Setup title and lecture lines
        title = "The Logic of the 'Empty Set'"
        lines = [
            "Imagine an empty magician's hat with nothing inside.",
            "This represents an empty set in mathematics.",
            "There is exactly one way to arrange nothing."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Imagine an empty magician's hat with nothing inside.
        # Apply color change to lecture line
        self.play(self.lecture[0].animate.set_color("#4B0082"))
        
        # Magician's hat (Represented by a Circle to avoid missing SVG asset error)
        hat = Circle(color="#4B0082")
        # Position hat in center of animation area
        self.place_in_area(hat, "B2", "E5", scale_factor=0.7)
        
        # Title text above the hat (Color #FFFFFF)
        top_text = Text("The Set of Nothing", font_size=24, color="#FFFFFF")
        # Fix for Issue 33: Adjusted area and scale
        self.place_in_area(top_text, "A1", "A6", scale_factor=0.8)
        
        self.play(FadeIn(hat), Write(top_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This represents an empty set in mathematics.
        # Apply color change to lecture line
        self.play(self.lecture[1].animate.set_color("#A9A9A9"))
        
        # Label 'Empty Set' (Color #A9A9A9)
        label = Text("Empty Set", font_size=30, color="#A9A9A9")
        # Fix for Issue 34: Place at specific grid point for precision
        self.place_at_grid(label, "D3", scale_factor=0.8)
        
        # Zoom into the empty hat effect (Visualizing the inside)
        self.play(
            hat.animate.scale(1.2),
            FadeIn(label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # There is exactly one way to arrange nothing.
        # Apply color change to lecture line
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Configuration text below the hat (Color #00FF00)
        bottom_text = Text("1 Configuration: Doing Nothing", font_size=24, color="#00FF00")
        # Fix for Issue 32: Rescaled and expanded area
        self.place_in_area(bottom_text, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(bottom_text))
        self.wait(2)
