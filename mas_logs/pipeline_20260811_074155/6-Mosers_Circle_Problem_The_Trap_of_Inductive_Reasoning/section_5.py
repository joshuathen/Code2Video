from manim import *
import itertools
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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Why did the simple doubling pattern fail?",
            "Every intersection inside the circle creates a new region.",
            "Four points are needed for each internal intersection.",
            "The formula uses combinations: C(n,4) plus C(n,2) plus one.",
            "Geometry reveals the truth where intuition failed."
        ]
        self.setup_layout("Cracking the Code: The Hidden Geometry", lecture_lines)

        # Colors
        COLOR_HIGHLIGHT = "#FFFF00"  # Yellow
        COLOR_FORMULA = "#FFFFFF"   # White
        COLOR_CIRCLE = "#888888"
        COLOR_CHORD = "#4444FF"

        # Helper for intersection
        def get_intersection(p1, p2, p3, p4):
            x1, y1 = p1[:2]
            x2, y2 = p2[:2]
            x3, y3 = p3[:2]
            x4, y4 = p4[:2]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6: return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return np.array([px, py, 0])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Circle and points for n=6
        circle_main = Circle(radius=1.8, color=COLOR_CIRCLE)
        # Issue 27: self.place_in_area(circle, 'C2', 'F5', scale_factor=0.8)
        self.place_in_area(circle_main, "C2", "F5", scale_factor=0.8)
        
        # Define points (avoiding regular hexagon to keep intersections clear)
        angles = [30, 85, 140, 210, 275, 330]
        points = [circle_main.point_at_angle(a * DEGREES) for a in angles]
        dots = VGroup(*[Dot(p, radius=0.06, color=WHITE) for p in points])
        
        # Create all chords
        chords = VGroup()
        for i, j in itertools.combinations(range(6), 2):
            chords.add(Line(points[i], points[j], stroke_width=2, color=COLOR_CHORD))
            
        self.play(Create(circle_main), Create(dots), Create(chords))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        intersections = []
        for combo in itertools.combinations(range(6), 4):
            idx = sorted(combo)
            p = get_intersection(points[idx[0]], points[idx[2]], points[idx[1]], points[idx[3]])
            if p is not None:
                intersections.append(p)
        
        intersection_dots = VGroup(*[Dot(p, radius=0.04, color=COLOR_HIGHLIGHT) for p in intersections])
        self.play(FadeIn(intersection_dots))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Focus on one intersection
        sample_indices = [0, 1, 3, 4]
        sample_points = VGroup(*[dots[i] for i in sample_indices])
        p1, p2, p3, p4 = points[0], points[3], points[1], points[4]
        sample_chords = VGroup(
            Line(p1, p2, color=YELLOW, stroke_width=4),
            Line(p3, p4, color=YELLOW, stroke_width=4)
        )
        sample_intersection = Dot(get_intersection(p1, p2, p3, p4), radius=0.08, color=YELLOW)
        
        self.play(
            intersection_dots.animate.set_opacity(0.2),
            chords.animate.set_stroke(opacity=0.2),
            FadeIn(sample_chords),
            sample_points.animate.scale(1.5).set_color(YELLOW),
            sample_intersection.animate.scale(1.5)
        )
        self.wait(2)
        
        self.play(
            FadeOut(sample_chords),
            sample_points.animate.scale(1/1.5).set_color(WHITE),
            sample_intersection.animate.scale(1/1.5).set_color(COLOR_HIGHLIGHT),
            intersection_dots.animate.set_opacity(1),
            chords.animate.set_stroke(opacity=1)
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Formula display
        formula_parts = MathTex(
            r"\text{Regions}", "=", r"\binom{n}{4}", "+", r"\binom{n}{2}", "+", "1",
            color=COLOR_FORMULA, font_size=32
        )
        # Issue 28: self.place_in_area(formula_parts, 'A2', 'A5', scale_factor=0.8)
        self.place_in_area(formula_parts, 'A2', 'A5', scale_factor=0.8)
        
        calc = MathTex(r"n=6 \implies 15 + 15 + 1 = 31", color=COLOR_FORMULA, font_size=28)
        # Issue 29: self.place_in_area(calc, 'B2', 'B5', scale_factor=0.8)
        self.place_in_area(calc, 'B2', 'B5', scale_factor=0.8)
        
        self.play(Write(formula_parts))
        self.wait(0.5)
        self.play(Write(calc))
        
        # Highlight the C(n,4) term
        self.play(formula_parts[2].animate.set_color(COLOR_HIGHLIGHT).scale(1.2))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        self.play(
            formula_parts[2].animate.set_color(COLOR_FORMULA).scale(1/1.2),
            circle_main.animate.set_stroke(color=YELLOW, width=4)
        )
        self.wait(3)
