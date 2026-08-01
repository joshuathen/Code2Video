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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Prerequisite Knowledge: Vertices and Intersections"
        lines = [
            "To maximize regions, avoid three lines meeting inside.",
            "Every internal intersection comes from exactly four points.",
            "Max offsets his threads to maximize the space."
        ]
        self.setup_layout(title, lines)
        
        # Helper to get points on a circle relative to its center
        def get_circle_point(angle_deg, radius=1.0):
            return np.array([radius * np.cos(np.radians(angle_deg)), radius * np.sin(np.radians(angle_deg)), 0])

        # === Animation for Lecture Line 1 ===
        # Color the first line
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Left circle (Sub-optimal: 3 lines crossing at center)
        bad_circle = Circle(radius=1.0, color=WHITE)
        bad_lines = VGroup(
            Line(get_circle_point(0), get_circle_point(180), color="#FF0000"),
            Line(get_circle_point(60), get_circle_point(240), color="#FF0000"),
            Line(get_circle_point(120), get_circle_point(300), color="#FF0000")
        )
        bad_group = VGroup(bad_circle, bad_lines)
        self.place_in_area(bad_group, "B1", "D3", scale_factor=0.8)
        bad_label = Text("Sub-optimal", font_size=20, color="#FF0000")
        self.place_in_area(bad_label, "E1", "E3", scale_factor=0.8)

        # Right circle (Optimal: Slightly offset)
        good_circle = Circle(radius=1.0, color=WHITE)
        good_lines = VGroup(
            Line(get_circle_point(10), get_circle_point(190), color="#00FF00"),
            Line(get_circle_point(70), get_circle_point(260), color="#00FF00"),
            Line(get_circle_point(130), get_circle_point(310), color="#00FF00")
        )
        good_group = VGroup(good_circle, good_lines)
        self.place_in_area(good_group, "B4", "D6", scale_factor=0.8)
        good_label = Text("Optimal", font_size=20, color="#00FF00")
        self.place_in_area(good_label, "E4", "E6", scale_factor=0.8)

        self.play(FadeIn(bad_group), FadeIn(bad_label), FadeIn(good_group), FadeIn(good_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Color the second line
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Remove bad group and transition good group to zoom view
        self.play(FadeOut(bad_group), FadeOut(bad_label), FadeOut(good_label))
        
        # Create a "Zoomed" version of the optimal intersection principle
        # Focus on the 'X' shape formed by 4 points
        zoom_circle = Circle(radius=1.5, color=WHITE)
        p1, p2, p3, p4 = 40, 140, 220, 320
        pts = [get_circle_point(a, radius=1.5) for a in [p1, p2, p3, p4]]
        
        boundary_dots = VGroup(*[Dot(p, color=YELLOW, radius=0.08) for p in pts])
        chord1 = Line(pts[0], pts[2], color="#00FF00")
        chord2 = Line(pts[1], pts[3], color="#00FF00")
        
        zoom_group = VGroup(zoom_circle, chord1, chord2, boundary_dots)
        self.place_in_area(zoom_group, "B2", "E5", scale_factor=1.0)
        
        # Highlight points one by one
        self.play(Transform(good_group, zoom_group))
        self.play(Create(boundary_dots), run_time=1)
        
        point_label = Text("4 Boundary Points = 1 Intersection", font_size=18, color=WHITE)
        self.place_in_area(point_label, "F2", "F5", scale_factor=1.0)
        self.play(Write(point_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Color the third line
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        rule_label = Text("Rule: No three lines may intersect at the same point", font_size=20, color="#FFFF00")
        # Position rule label at the top of the right-side grid area across row A
        self.place_in_area(rule_label, "A1", "A6", scale_factor=0.9)
        
        self.play(FadeIn(rule_label))
        self.wait(2)
