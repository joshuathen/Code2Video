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
        # Initializing the layout with parameters from the section brief
        title_text = "The Strange Encounter"
        lecture_lines = [
            "Geometry's foundation is the circle and the constant pi.",
            "Prime numbers are the discrete atoms of arithmetic.",
            "How do smooth circles relate to jagged prime numbers?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors based on visual anchor system instructions
        COLOR_PI = "#FFD700"
        COLOR_PRIME = "#00FFFF"
        COLOR_RELATION = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Focusing on the foundation: Circle and pi
        self.play(self.lecture[0].animate.set_color(COLOR_PI))
        
        # Circle with radius r and pi at the center
        circle = Circle(radius=1.2, color=COLOR_PI)
        pi_sym = Text("\u03c0", color=COLOR_PI, font_size=56)
        radius_line = Line(circle.get_center(), circle.get_right(), color=COLOR_PI, stroke_width=2)
        r_label = Text("r", color=COLOR_PI, font_size=20).next_to(radius_line, UP, buff=0.1)
        
        circle_container = VGroup(circle, pi_sym, radius_line, r_label)
        # Fix for Issue #26: Positioning and scaling to avoid crowding title
        self.place_in_area(circle_container, "B2", "C5", scale_factor=0.8)
        
        self.play(FadeIn(circle_container))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Prime numbers: The discrete atoms
        self.play(self.lecture[1].animate.set_color(COLOR_PRIME))
        
        primes_list = Text("2, 3, 5, 7, 11, 13, 17, 19...", color=COLOR_PRIME, font_size=32)
        # Fix for Issue #27: Reduced scale to improve visual spacing
        self.place_in_area(primes_list, "E1", "F6", scale_factor=0.7)
        
        self.play(FadeIn(primes_list))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The link: Smooth circles vs Jagged numbers
        self.play(self.lecture[2].animate.set_color(COLOR_RELATION))
        
        # Glowing arrow connecting primes to pi
        link_arrow = Arrow(
            start=primes_list.get_top(),
            end=circle_container.get_bottom(),
            color=COLOR_RELATION,
            stroke_width=6,
            buff=0.3
        )
        
        question_mark = Text("?", color=COLOR_RELATION, font_size=80)
        # Fix for Issue #28: Positioning in area for better vertical alignment
        self.place_in_area(question_mark, "D3", "D4", scale_factor=0.8)
        
        self.play(
            GrowArrow(link_arrow),
            FadeIn(question_mark)
        )
        self.wait(3)
