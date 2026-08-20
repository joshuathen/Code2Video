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
        lecture_lines = ["Should we guess sixteen?", "Let's count five points carefully.", "The result is only sixteen regions."]
        self.setup_layout("Prerequisite: Combinatorics Basics", lecture_lines)
        
        # Define geometry
        circle = Circle(radius=1.5, color=WHITE)
        points = [
            circle.point_from_proportion(i / 5) for i in range(5)
        ]
        dots = VGroup(*[Dot(point, color=YELLOW) for point in points])
        
        # Setup visual elements
        self.place_in_area(circle, 'B3', 'E4', scale_factor=0.6)
        for dot in dots:
            # Need to re-scale the dots because they were positioned using relative circle points 
            # and circle was scaled down.
            dot.move_to(circle.point_from_proportion(dots.submobjects.index(dot) / 5))
            self.add(dot)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Highlight points
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # Draw intersecting lines (e.g., connect point 0 to 2 and 1 to 3)
        # Choosing 4 points (0, 1, 2, 3) to form an intersection
        line1 = Line(points[0], points[2], color=RED)
        line2 = Line(points[1], points[3], color=RED)
        intersection_dot = Dot(color=GREEN)
        
        self.play(Create(line1), Create(line2))
        self.place_at_grid(intersection_dot, 'C3', scale_factor=0.5)
        self.play(FadeIn(intersection_dot))
        
        self.wait(2)
