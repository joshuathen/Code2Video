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
        self.setup_layout("Geometric Logic: Counting V and E", [
            "Vertices include n boundary points and internal intersections.",
            "Every 4 points create one internal intersection.",
            "Total vertices V equals n plus nC4.",
            "Edges connect vertices along chords and arcs.",
            "Total edges E equals n plus nC2 plus 2nC4."
        ])

        # Helper function for intersection
        def get_line_intersection(p1, p2, p3, p4):
            x1, y1 = p1[:2]
            x2, y2 = p2[:2]
            x3, y3 = p3[:2]
            x4, y4 = p4[:2]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if abs(denom) < 1e-6: return None
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1), 0])

        # Initial Setup
        circle = Circle(radius=2.0, color=WHITE)
        self.place_in_area(circle, "B2", "E5")
        
        # We use n = 6 for demonstration
        n = 6
        phi = np.linspace(0, 2*np.pi, n, endpoint=False)
        phi += 0.2 # Offset for better intersection visibility
        points = [circle.point_at_angle(p) for p in phi]
        boundary_dots = VGroup(*[Dot(p, color=WHITE, radius=0.08) for p in points])
        
        # All possible chords
        chords = VGroup()
        for i in range(n):
            for j in range(i + 1, n):
                chords.add(Line(points[i], points[j], stroke_width=1, color=GREY_D))

        # === Animation for Lecture Line 1 ===
        # Vertices include n boundary points and internal intersections.
        self.lecture[0].set_color(WHITE)
        
        self.play(Create(circle))
        self.play(FadeIn(boundary_dots))
        self.play(Create(chords))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Every 4 points create one internal intersection.
        # Storyboard: Highlight 4 points in magenta, intersection in cyan.
        self.lecture[0].set_color(GREY_B)
        self.lecture[1].set_color("#00FFFF") # Matches cyan intersection
        
        indices = [0, 1, 3, 4]
        selected_points = [points[i] for i in indices]
        mag_dots = VGroup(*[Dot(p, color="#FF00FF", radius=0.1) for p in selected_points])
        
        # Crossing chords: 0-3 and 1-4
        c1 = Line(selected_points[0], selected_points[2], color="#00FFFF", stroke_width=3)
        c2 = Line(selected_points[1], selected_points[3], color="#00FFFF", stroke_width=3)
        
        intersect_point = get_line_intersection(selected_points[0], selected_points[2], 
                                               selected_points[1], selected_points[3])
        int_dot = Dot(intersect_point, color="#00FFFF", radius=0.12)
        
        self.play(FadeIn(mag_dots))
        self.play(Create(c1), Create(c2))
        self.play(Flash(int_dot, color="#00FFFF"), FadeIn(int_dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Total vertices V equals n plus nC4.
        # Storyboard: Display V formula in yellow.
        self.lecture[1].set_color(GREY_B)
        self.lecture[2].set_color("#FFFF00")
        
        v_formula = MathTex("V = n + \\binom{n}{4}", color="#FFFF00")
        # Resolution of Issue 33: Use place_in_area for better centering and scale
        self.place_in_area(v_formula, 'A4', 'A6', scale_factor=0.9)
        
        self.play(Write(v_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Edges connect vertices along chords and arcs.
        self.lecture[2].set_color(GREY_B)
        self.lecture[3].set_color("#00FFFF")
        
        # Highlight boundary arcs
        arcs = VGroup(*[Arc(radius=2.0, start_angle=phi[i], angle=(phi[(i+1)%n]-phi[i]) % (2*np.pi), 
                            color="#00FFFF", stroke_width=4).move_to(circle.get_center()) for i in range(n)])
        
        self.play(Create(arcs))
        self.play(chords.animate.set_color("#00FFFF").set_stroke(width=1.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Total edges E equals n plus nC2 plus 2nC4.
        # Storyboard: Display E formula in cyan.
        self.lecture[3].set_color(GREY_B)
        self.lecture[4].set_color("#00FFFF")
        
        e_formula = MathTex("E = n + \\binom{n}{2} + 2\\binom{n}{4}", color="#00FFFF")
        # Resolution of Issue 34: Use place_in_area for long formula and adjusted scale
        self.place_in_area(e_formula, 'F3', 'F6', scale_factor=0.8)
        
        self.play(Write(e_formula))
        self.wait(2)

# Marking issues as resolved
# update_issue(33, under_review=True, resolution_note="Used place_in_area(v_formula, 'A4', 'A6', scale_factor=0.9) to improve centering and readability.")
# update_issue(34, under_review=True, resolution_note="Used place_in_area(e_formula, 'F3', 'F6', scale_factor=0.8) to accommodate the length of the formula.")
