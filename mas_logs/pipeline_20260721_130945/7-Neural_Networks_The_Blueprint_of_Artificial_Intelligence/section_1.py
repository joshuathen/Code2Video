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
        # 1. Setup Layout
        title_text = "The Biological Inspiration: The Puppy Analogy"
        lecture_lines = [
            "Neural networks learn from experience, much like a puppy.",
            "Meet Pixel, a digital puppy learning to recognize objects.",
            "Pixel uses color and shape to distinguish balls from lemons."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets - Replaced SVGMobjects with standard Mobjects to avoid file path errors
        pixel_puppy = Triangle(fill_opacity=1.0)
        ball = Circle(fill_opacity=1.0)
        lemon = Ellipse(width=1.0, height=0.6, fill_opacity=1.0)

        # Color assets for clarity
        ball.set_color(RED)
        lemon.set_color(YELLOW)
        pixel_puppy.set_color(ORANGE)

        # === Animation for Lecture Line 1 ===
        # Positioning placeholders relative to the grid
        self.place_in_area(pixel_puppy, 'B1', 'E3', scale_factor=0.8)
        self.place_at_grid(ball, 'B5', scale_factor=0.5)
        self.place_at_grid(lemon, 'E5', scale_factor=0.5)

        self.play(
            FadeIn(pixel_puppy),
            FadeIn(ball),
            FadeIn(lemon),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ORANGE))
        self.play(Indicate(pixel_puppy))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Display text 'Input: Yellow + Round' near the lemon object
        input_text = Text("Input: Yellow + Round", font_size=18, color=YELLOW)
        self.place_at_grid(input_text, 'F5', scale_factor=0.8)
        
        self.play(Write(input_text))
        self.wait(1)

        # Animate 'Pixel' looking at lemon with a '?' appearing above
        q_mark = Text("?", font_size=36, color=WHITE)
        self.place_at_grid(q_mark, 'A2', scale_factor=1.0)
        
        self.play(
            pixel_puppy.animate.rotate(10 * DEGREES),
            FadeIn(q_mark)
        )
        self.play(pixel_puppy.animate.rotate(-10 * DEGREES))
        self.wait(1)

        # Label lemon as 'Snack' and ball as 'Toy'
        snack_label = Text("Snack", font_size=20, color="#00FF00")
        toy_label = Text("Toy", font_size=20, color="#0000FF")
        
        self.place_at_grid(snack_label, 'E6', scale_factor=0.8)
        self.place_at_grid(toy_label, 'B6', scale_factor=0.8)
        
        self.play(Write(snack_label), Write(toy_label))
        self.wait(1)

        # Show 'Pixel' puppy reacting after identification
        self.play(FadeOut(q_mark), FadeOut(input_text))
        self.play(Indicate(pixel_puppy, color=PINK))
        self.play(Indicate(ball, color=BLUE))
        self.wait(2)
