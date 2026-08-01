from manim import *
import numpy as np

# Helper function for calculating intersection of two lines defined by points
def get_line_intersection(p1, p2, p3, p4):
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]
    x4, y4 = p4[:2]
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0:
        return np.array([0, 0, 0])
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1), 0])

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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Regions depend on internal intersections and boundary lines.",
            "Four points create exactly one intersection point.",
            "The formula is C(n, 4) plus C(n, 2) plus 1.",
            "For six points, this sum equals thirty-one.",
            "Math explains why the simple pattern was wrong."
        ]
        self.setup_layout("The True Mathematical Formula", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Display the formula in a white framed box
        formula = Text(
            "Regions = C(n, 4) + C(n, 2) + 1",
            color=WHITE, font_size=36
        )
        formula_box = SurroundingRectangle(formula, color=WHITE, buff=0.2)
        formula_group = VGroup(formula_box, formula)
        self.place_in_area(formula_group, 'A1', 'B6', scale_factor=0.9)
        
        self.play(Create(formula_box), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        circle1 = Circle(radius=0.8, color=BLUE_E)
        points1 = [circle1.point_at_angle(i * TAU / 6) for i in range(6)]
        dots1 = VGroup(*[Dot(p, radius=0.05, color=GRAY) for p in points1])
        
        # Selection of 4 points on the circle defining 1 internal intersection
        highlight_indices = [0, 1, 3, 4]
        highlights1 = VGroup(*[Dot(points1[i], radius=0.08, color=YELLOW) for i in highlight_indices])
        
        chord1 = Line(points1[0], points1[3], stroke_width=2, color=WHITE)
        chord2 = Line(points1[1], points1[4], stroke_width=2, color=WHITE)
        
        intersect_pt = get_line_intersection(points1[0], points1[3], points1[1], points1[4])
        intersect_dot = Dot(intersect_pt, radius=0.06, color=RED)
        
        vis1 = VGroup(circle1, dots1, highlights1, chord1, chord2, intersect_dot)
        self.place_in_area(vis1, 'C1', 'D3', scale_factor=1.0)
        
        label1 = Text("C(n, 4)", font_size=28).next_to(vis1, DOWN, buff=0.1)
        
        self.play(Create(circle1), FadeIn(dots1))
        self.play(FadeIn(highlights1))
        self.play(Create(chord1), Create(chord2))
        self.play(FadeIn(intersect_dot), Write(label1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        circle2 = Circle(radius=0.8, color=BLUE_E)
        points2 = [circle2.point_at_angle(i * TAU / 6) for i in range(6)]
        dots2 = VGroup(*[Dot(p, radius=0.05, color=GRAY) for p in points2])
        
        # Selection of 2 points on the circle defining 1 chord
        highlight_indices2 = [0, 2]
        highlights2 = VGroup(*[Dot(points2[i], radius=0.08, color=YELLOW) for i in highlight_indices2])
        
        chord3 = Line(points2[0], points2[2], stroke_width=2, color=WHITE)
        
        vis2 = VGroup(circle2, dots2, highlights2, chord3)
        self.place_in_area(vis2, 'C4', 'D6', scale_factor=1.0)
        
        label2 = Text("C(n, 2)", font_size=28).next_to(vis2, DOWN, buff=0.1)
        
        self.play(Create(circle2), FadeIn(dots2))
        self.play(FadeIn(highlights2))
        self.play(Create(chord3))
        self.play(Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Substitution formula for n=6
        substitution = Text(
            "C(6, 4) + C(6, 2) + 1 = 15 + 15 + 1 = 31",
            font_size=28
        )
        # Issue 42: Optimized placement to avoid crowding lecture lines
        self.place_in_area(substitution, 'E2', 'F6', scale_factor=0.8)
        
        self.play(Write(substitution))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Split equation to isolate '31' for final animation
        sub_parts = VGroup(
            Text("C(6, 4) + C(6, 2) + 1 = 15 + 15 + 1 = ", font_size=28),
            Text("31", font_size=28)
        ).arrange(RIGHT, buff=0.1)
        
        # Issue 43: Adjusted placement to avoid overlap with visualizations
        self.place_at_grid(sub_parts, 'F6', scale_factor=0.9)
        
        self.remove(substitution)
        self.add(sub_parts)
        
        res_31 = sub_parts[1]
        
        # Issue 33: Load based icon asset
        based_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/based.svg")
        
        # '31' glows bright green
        self.play(res_31.animate.set_color("#00FF00").scale(1.5))
        self.play(Indicate(res_31, color="#00FF00", scale_factor=1.2))
        
        # Result '31' moves to the center next to the based icon
        final_group = VGroup(res_31, based_icon).arrange(RIGHT, buff=0.4)
        self.play(
            final_group.animate.move_to(ORIGIN).scale(1.5),
            FadeIn(based_icon)
        )
        self.wait(2)
