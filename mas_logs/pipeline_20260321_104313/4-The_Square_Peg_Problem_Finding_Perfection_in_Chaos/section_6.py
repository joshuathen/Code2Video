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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and the 5 mandatory lecture lines
        self.setup_layout("The Twist: The Möbius Strip Reveal", [
            "Swapping points A and B represents the same pair.",
            "This symmetry requires us to glue the grid's edges.",
            "The grid twists into a topological Möbius strip.",
            "This strange shape has only one single boundary.",
            "It perfectly captures the geometry of point pairs."
        ])

        # === Animation for Lecture Line 1 ===
        # Two points on a circle swap labels (A to B, B to A)
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Geometry for circle visualization
        circle = Circle(radius=1.5, color=WHITE)
        dot_a = Dot(circle.point_at_angle(PI/4), color="#FF0000") # Red
        dot_b = Dot(circle.point_at_angle(5*PI/4), color="#0000FF") # Blue
        
        # Relative positions for labels
        p_a = circle.point_at_angle(PI/4) + UR*0.3
        p_b = circle.point_at_angle(5*PI/4) + DL*0.3
        
        label_a = Text("A", font_size=24, color=WHITE).shift(p_a)
        label_b = Text("B", font_size=24, color=WHITE).shift(p_b)
        line_ab = Line(dot_a.get_center(), dot_b.get_center(), color=GREY, stroke_opacity=0.5)
        
        circle_group = VGroup(circle, dot_a, dot_b, label_a, label_b, line_ab)
        # Use visual anchor system to place on the right side
        self.place_in_area(circle_group, "B2", "E6", scale_factor=0.7)
        
        self.play(Create(circle), Create(line_ab))
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        
        # Fetch current centers after group placement for precise swapping
        target_a_pos = label_a.get_center()
        target_b_pos = label_b.get_center()
        
        label_a_swap = Text("B", font_size=24, color=WHITE).move_to(target_a_pos)
        label_b_swap = Text("A", font_size=24, color=WHITE).move_to(target_b_pos)
        
        # Animate the identity swap (A,B) = (B,A)
        self.play(
            ReplacementTransform(label_a, label_a_swap),
            ReplacementTransform(label_b, label_b_swap),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A rectangle shows arrows on edges indicating a topological 'twist' glueing
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create a flat grid representing the configuration space
        rect_w, rect_h = 4.0, 3.0
        base_rect = Rectangle(width=rect_w, height=rect_h, color=WHITE)
        h_lines = VGroup(*[
            Line(LEFT*rect_w/2, RIGHT*rect_w/2, color=WHITE, stroke_width=1) 
            for _ in range(4)
        ]).arrange(DOWN, buff=rect_h/3)
        v_lines = VGroup(*[
            Line(UP*rect_h/2, DOWN*rect_h/2, color=WHITE, stroke_width=1) 
            for _ in range(5)
        ]).arrange(RIGHT, buff=rect_w/4)
        
        flat_grid = VGroup(base_rect, h_lines, v_lines)
        
        # ISSUE 53: Position the flat_grid in 'B2' to 'E6' for balance
        self.place_in_area(flat_grid, "B2", "E6", scale_factor=0.8)
        
        # Identification arrows for Möbius twist
        arrow_left = Arrow(
            start=flat_grid.get_left() + DOWN*0.7, 
            end=flat_grid.get_left() + UP*0.7, 
            color="#58C4DD", stroke_width=6, buff=0
        )
        arrow_right = Arrow(
            start=flat_grid.get_right() + UP*0.7, 
            end=flat_grid.get_right() + DOWN*0.7, 
            color="#58C4DD", stroke_width=6, buff=0
        )
        
        # Transition from circle example to the topology grid
        self.play(FadeOut(circle_group), FadeOut(label_a_swap), FadeOut(label_b_swap))
        self.play(Create(flat_grid))
        self.play(GrowArrow(arrow_left), GrowArrow(arrow_right))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The grid twists into a topological Möbius strip.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Generate Möbius strip segments for a 2D projection
        mobius_lines = VGroup()
        segments = 60
        radius = 1.4
        width = 0.7
        for i in range(segments):
            theta = (i / segments) * 2 * PI
            cx, cy = radius * np.cos(theta), radius * np.sin(theta)
            # Möbius strip parametrization
            tx = width * np.cos(theta / 2) * np.cos(theta)
            ty = width * np.cos(theta / 2) * np.sin(theta)
            tz = width * np.sin(theta / 2) * 0.5 # Perspective shift
            p1 = np.array([cx + tx, cy + ty + tz, 0])
            p2 = np.array([cx - tx, cy - ty - tz, 0])
            mobius_lines.add(Line(p1, p2, color=WHITE, stroke_opacity=0.5, stroke_width=1.5))

        # ISSUE 54: Position the mobius_lines in 'B2' to 'E6' with reduced scale
        self.place_in_area(mobius_lines, "B2", "E6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(flat_grid, mobius_lines),
            FadeOut(arrow_left), FadeOut(arrow_right)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Trace the single boundary of the Möbius strip
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # The boundary of a Möbius strip is a single closed curve
        boundary = ParametricFunction(
            lambda t: np.array([
                (radius + width * np.cos(t/2)) * np.cos(t),
                (radius + width * np.cos(t/2)) * np.sin(t) + (width * np.sin(t/2) * 0.5),
                0
            ]),
            t_range=[0, 4*PI],
            color="#00FFFF",
            stroke_width=5
        )
        # ISSUE 54: Position the boundary in 'B2' to 'E6'
        self.place_in_area(boundary, "B2", "E6", scale_factor=0.8)
        
        self.play(Create(boundary), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It perfectly captures the geometry of point pairs.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        label = Text("Pairs (A,B) = (B,A)", font_size=28, color=WHITE)
        # ISSUE 52: Centered label across F3-F5 to prevent clipping
        self.place_in_area(label, "F3", "F5", scale_factor=0.7)
        
        self.play(Write(label))
        self.wait(2)
