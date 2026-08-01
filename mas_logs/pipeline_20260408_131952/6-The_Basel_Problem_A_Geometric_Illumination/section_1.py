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
        # Initializing the layout with the title and lecture notes
        title = "The Infinite Challenge"
        lines = [
            "Imagine an infinite row of glowing light bulbs.",
            "Each bulb sits at a whole number distance away.",
            "Their total brightness forms a famous mathematical mystery.",
            "For ninety years, this sum eluded the greatest minds.",
            "Until Leonhard Euler discovered a shocking geometric link."
        ]
        self.setup_layout(title, lines)

        # Transition title and lecture into view
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # Multiple light bulbs (#E0FFFF) fade in along a horizontal line.
        # Resolves Issue 30, 31, 32
        self.play(self.lecture[0].animate.set_color("#E0FFFF"))
        
        bulbs = VGroup()
        labels = VGroup()
        
        # Fixing Bulb 1 at C1 with scale 1.2 and color #E0FFFF (Issue 30, 32)
        circle = Circle(radius=0.4, color="#E0FFFF", fill_opacity=0.8)
        self.place_at_grid(circle, 'C1', scale_factor=1.2)
        
        # Fixing label at C1 with scale 0.8 (Issue 31)
        point_label = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(point_label, 'C1', scale_factor=0.8)
        
        bulbs.add(circle)
        labels.add(point_label)
        
        # Generate rest of the row
        for i in range(2, 7):
            pos = f"C{i}"
            b = Circle(radius=0.4, color="#E0FFFF", fill_opacity=0.6)
            self.place_at_grid(b, pos, scale_factor=1.0 - (i*0.1))
            l = Text(str(i), font_size=24, color=WHITE)
            self.place_at_grid(l, pos, scale_factor=0.7 - (i*0.05))
            bulbs.add(b)
            labels.add(l)
            
        self.play(FadeIn(bulbs), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pixel the Robot (#00FF00) appears at the origin.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Simple representation of Pixel the Robot
        pixel_body = Square(side_length=0.4, fill_opacity=1, color="#00FF00")
        pixel_head = Circle(radius=0.12, fill_opacity=1, color="#00FF00")
        pixel = VGroup(pixel_body, pixel_head).arrange(UP, buff=0.05)
        # Position him at 'D1' (just below/near the first bulb)
        self.place_at_grid(pixel, "D1", scale_factor=1.0)
        
        self.play(FadeIn(pixel, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use Text instead of MathTex to avoid FileNotFoundError if LaTeX is not installed
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        mystery_expr = Text("1 + 1/4 + 1/9 + 1/16 + ...", color="#FFFFFF")
        # Center in the B row area
        self.place_in_area(mystery_expr, "B2", "B6", scale_factor=1.0)
        
        self.play(Write(mystery_expr))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A pulsing red question mark (#FF0000) appears.
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        
        q_mark = Text("?", color="#FF0000", font_size=60)
        self.place_at_grid(q_mark, "A4", scale_factor=1.2)
        
        self.play(FadeIn(q_mark))
        self.play(q_mark.animate.scale(1.2), rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Leonhard Euler (#00FFFF) appears with a circular outline (#FFFFFF).
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        
        euler_name = Text("Leonhard Euler", color="#00FFFF", font_size=28)
        self.place_at_grid(euler_name, "E4", scale_factor=1.0)
        
        euler_circle = Circle(color=WHITE)
        self.place_at_grid(euler_circle, "E4", scale_factor=2.0)
        
        self.play(Write(euler_name), Create(euler_circle))
        self.wait(2)
