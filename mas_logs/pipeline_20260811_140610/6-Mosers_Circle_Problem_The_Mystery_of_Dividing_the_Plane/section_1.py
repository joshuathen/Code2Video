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
        self.setup_layout("The Hook: How many pieces?", ["Place n points on a circle.", "Connect every pair with chords.", "How many regions are created?"])
        
        # Setup assets/circle
        # Using Circle instead of SVGMobject to ensure it has path data for point_from_proportion
        circle_container = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle_container, 'B2', 'E5', scale_factor=1.0)
        circle_container.set_color("#FFFFFF")
        
        # n = 6 points
        n = 6
        points = [Dot(circle_container.point_from_proportion(i / n), color=WHITE) for i in range(n)]
        point_group = VGroup(*points)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(circle_container), FadeIn(point_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        chords = VGroup()
        for i in range(n):
            for j in range(i + 1, n):
                chords.add(Line(points[i].get_center(), points[j].get_center(), color="#FFD700", stroke_width=2))
        
        self.play(Create(chords))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Visual highlighting of regions
        # Simplified: highlight the center region
        region_highlight = Polygon(
            points[0].get_center(), points[2].get_center(), points[4].get_center(),
            color="#00FF00", fill_opacity=0.3
        )
        self.play(FadeIn(region_highlight))
        self.wait(1)
