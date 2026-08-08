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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mathematical Anatomy", [
            "Vertices include original points and every internal crossing.",
            "Every four points define exactly one unique intersection.",
            "Total internal vertices are calculated as \"n choose 4\".",
            "Edges are chords plus segments along the circumference.",
            "We use combinations to find all parts of the puzzle."
        ])

        # Color definitions
        MAGENTA_CLR = "#FF00FF"
        CYAN_CLR = "#00FFFF"
        WHITE_CLR = "#FFFFFF"
        HIGHLIGHT_CLR = "#FFFF00"

        # Points setup for n=6 (using non-uniform angles to avoid triple intersections)
        n = 6
        angles = [0.1, 1.2, 2.3, 3.1, 4.4, 5.6]
        radius = 1.8
        
        # Centering the circle in the grid area B2 to E5
        tl = self.grid["B2"]
        br = self.grid["E5"]
        circle_center = (tl + br) / 2
        
        circle = Circle(radius=radius, color=WHITE_CLR).move_to(circle_center)
        pts = [circle_center + np.array([radius * np.cos(a), radius * np.sin(a), 0]) for a in angles]
        dot_mobjects = VGroup(*[Dot(p, color=MAGENTA_CLR, radius=0.08) for p in pts])
        
        # All chords
        chords = VGroup()
        for i in range(n):
            for j in range(i + 1, n):
                chords.add(Line(pts[i], pts[j], stroke_width=1, color=WHITE_CLR, stroke_opacity=0.5))

        # Helper function for intersection of two lines segments
        def get_intersection(p1, p2, p3, p4):
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]
            x4, y4 = p4[0], p4[1]
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6: return None
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            # Check if intersection is within segment bounds
            if (min(x1,x2)-1e-6 <= px <= max(x1,x2)+1e-6 and 
                min(y1,y2)-1e-6 <= py <= max(y1,y2)+1e-6 and
                min(x3,x4)-1e-6 <= px <= max(x3,x4)+1e-6 and
                min(y3,y4)-1e-6 <= py <= max(y3,y4)+1e-6):
                return np.array([px, py, 0])
            return None

        # Gather all unique internal intersection points
        intersection_points = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(n):
                    for l in range(k+1, n):
                        indices = {i, j, k, l}
                        if len(indices) == 4:
                            p = get_intersection(pts[i], pts[j], pts[k], pts[l])
                            if p is not None:
                                is_dup = False
                                for existing in intersection_points:
                                    if np.linalg.norm(existing - p) < 1e-4:
                                        is_dup = True
                                        break
                                if not is_dup:
                                    intersection_points.append(p)
        
        intersection_dots = VGroup(*[Dot(p, color=CYAN_CLR, radius=0.06) for p in intersection_points])

        # === Animation for Lecture Line 1 ===
        # Vertices include original points and every internal crossing.
        self.lecture[0].set_color(MAGENTA_CLR)
        self.play(Create(circle))
        self.play(Create(dot_mobjects))
        self.play(Create(chords), run_time=1.5)
        self.play(Create(intersection_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Every four points define exactly one unique intersection.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CYAN_CLR)
        
        # Highlight a specific set of 4 points and their intersection
        set1_indices = [0, 1, 2, 4]
        set1_dots = VGroup(*[dot_mobjects[i].copy().scale(1.5).set_color(HIGHLIGHT_CLR) for i in set1_indices])
        # Chords connecting them that intersect: (0,2) and (1,4)
        c1 = Line(pts[0], pts[2], color=HIGHLIGHT_CLR, stroke_width=3)
        c2 = Line(pts[1], pts[4], color=HIGHLIGHT_CLR, stroke_width=3)
        inter_pt = get_intersection(pts[0], pts[2], pts[1], pts[4])
        inter_dot = Dot(inter_pt, color=CYAN_CLR, radius=0.12)
        
        self.play(FadeIn(set1_dots))
        self.play(Create(c1), Create(c2))
        self.play(Flash(inter_dot, color=CYAN_CLR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Total internal vertices are calculated as "n choose 4".
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CYAN_CLR)
        
        formula_ncr = MathTex(r"\text{Internal Vertices} = \binom{n}{4}", color=CYAN_CLR)
        # Fix for Issue 33: Better placement in grid
        self.place_in_area(formula_ncr, 'A4', 'B6', scale_factor=0.8)
        
        self.play(Write(formula_ncr))
        
        # Highlight another set of 4 points
        set2_indices = [1, 2, 3, 5]
        set2_dots = VGroup(*[dot_mobjects[i].copy().scale(1.5).set_color(HIGHLIGHT_CLR) for i in set2_indices])
        # (1,3) and (2,5)
        c3 = Line(pts[1], pts[3], color=HIGHLIGHT_CLR, stroke_width=3)
        c4 = Line(pts[2], pts[5], color=HIGHLIGHT_CLR, stroke_width=3)
        
        self.play(FadeOut(set1_dots), FadeOut(c1), FadeOut(c2))
        self.play(FadeIn(set2_dots))
        self.play(Create(c3), Create(c4))
        self.wait(1)
        self.play(FadeOut(set2_dots), FadeOut(c3), FadeOut(c4))

        # === Animation for Lecture Line 4 ===
        # Edges are chords plus segments along the circumference.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_CLR)
        
        # Define circumference segments (arcs)
        arcs = VGroup()
        for i in range(n):
            start = angles[i]
            end = angles[(i+1)%n]
            if end < start: end += 2*PI
            arcs.add(Arc(radius=radius, start_angle=start, angle=end-start, color=HIGHLIGHT_CLR, stroke_width=4).move_to(circle_center))
        
        self.play(Create(arcs))
        self.play(chords.animate.set_stroke(opacity=1, color=HIGHLIGHT_CLR), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We use combinations to find all parts of the puzzle.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        summary_v = MathTex(r"V_{\text{total}} = n + \binom{n}{4}", color=WHITE)
        summary_e = MathTex(r"E_{\text{total}} = 2\binom{n}{4} + \binom{n}{2} + n", color=WHITE)
        summary_group = VGroup(summary_v, summary_e).arrange(DOWN, buff=0.3)
        # Fix for Issue 32: Better placement in grid
        self.place_in_area(summary_group, 'E4', 'F6', scale_factor=0.8)
        
        self.play(FadeOut(formula_ncr))
        self.play(Write(summary_group))
        self.wait(2)
