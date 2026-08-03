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

class Section4Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_str = "The 'Aha!' Moment: The Geometric Proof"
        lecture_lines = [
            "Pick any point on the elliptical slice.",
            "Distance to focus equals distance to the tangent circle.",
            "Sum of distances equals the segment between tangent circles.",
            "This segment length is constant along the cone's surface.",
            "Thus, PF1 plus PF2 is always constant."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        ELLIPSE_COLOR = "#FF4500" # Orange-Red
        F1_COLOR = "#FF69B4"      # Hot Pink
        F2_COLOR = "#1E90FF"      # Dodger Blue
        CONE_COLOR = GREY_B
        POINT_COLOR = WHITE

        # --- Geometry Setup ---
        # Cone Apex and base for side view
        apex = (self.grid["A3"] + self.grid["A4"]) / 2
        left_base = self.grid["F1"] + RIGHT * 0.2
        right_base = self.grid["F6"] - RIGHT * 0.2
        
        cone_left = Line(apex, left_base, color=CONE_COLOR)
        cone_right = Line(apex, right_base, color=CONE_COLOR)
        
        # Ellipse line (side view projection)
        e_left = self.grid["C2"] + RIGHT * 0.3
        e_right = self.grid["E5"] + LEFT * 0.3
        ellipse_line = Line(e_left, e_right, color=ELLIPSE_COLOR, stroke_width=4)
        
        # Foci F1 and F2 (points where spheres touch ellipse)
        f1_pos = ellipse_line.point_from_proportion(0.25)
        f2_pos = ellipse_line.point_from_proportion(0.75)
        f1 = Dot(f1_pos, color=F1_COLOR)
        f2 = Dot(f2_pos, color=F2_COLOR)
        f1_label = MathTex("F_1", color=F1_COLOR, font_size=20).next_to(f1, UP, buff=0.1)
        f2_label = MathTex("F_2", color=F2_COLOR, font_size=20).next_to(f2, DOWN, buff=0.1)

        # Dandelin Spheres (represented as circles)
        # sphere 1 (small, upper)
        s1_center = self.grid["B3"] + RIGHT * 0.4
        s1_radius = 0.55
        sphere1 = Circle(radius=s1_radius, color=WHITE, stroke_opacity=0.3).move_to(s1_center)
        
        # sphere 2 (large, lower)
        s2_center = self.grid["E4"] + LEFT * 0.4 + DOWN * 0.2
        s2_radius = 1.15
        sphere2 = Circle(radius=s2_radius, color=WHITE, stroke_opacity=0.3).move_to(s2_center)

        # Tangency Heights (Q1 and Q2 levels)
        t1_y = s1_center[1]
        t2_y = s2_center[1]

        # Helper functions for dynamic segments
        p_val = ValueTracker(0.5)
        
        def get_p_pos():
            return ellipse_line.point_from_proportion(p_val.get_value())

        def get_q_pos(y_level):
            p_pos = get_p_pos()
            gen_dir = (p_pos - apex)
            if abs(gen_dir[1]) < 1e-6: return p_pos # Avoid division by zero
            t = (y_level - apex[1]) / gen_dir[1]
            return apex + t * gen_dir

        # Updateable elements
        p_dot = Dot(color=POINT_COLOR).add_updater(lambda d: d.move_to(get_p_pos()))
        p_label = MathTex("P", color=POINT_COLOR, font_size=20).add_updater(lambda l: l.next_to(p_dot, UR, buff=0.05))
        
        # PF1 and PF2 lines
        pf1 = Line(get_p_pos(), f1_pos, color=F1_COLOR)
        pf1.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f1_pos))
        
        pf2 = Line(get_p_pos(), f2_pos, color=F2_COLOR)
        pf2.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f2_pos))
        
        # PQ1 and PQ2 lines (on the surface of the cone along generator)
        pq1 = Line(get_p_pos(), get_q_pos(t1_y), color=F1_COLOR)
        pq1.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), get_q_pos(t1_y)))
        
        pq2 = Line(get_p_pos(), get_q_pos(t2_y), color=F2_COLOR)
        pq2.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), get_q_pos(t2_y)))

        q1_dot = Dot(color=F1_COLOR, radius=0.04).add_updater(lambda d: d.move_to(get_q_pos(t1_y)))
        q2_dot = Dot(color=F2_COLOR, radius=0.04).add_updater(lambda d: d.move_to(get_q_pos(t2_y)))
        
        q1_label = MathTex("Q_1", color=F1_COLOR, font_size=18).add_updater(lambda l: l.next_to(q1_dot, LEFT, buff=0.1))
        q2_label = MathTex("Q_2", color=F2_COLOR, font_size=18).add_updater(lambda l: l.next_to(q2_dot, LEFT, buff=0.1))

        # === Animation for Lecture Line 1 ===
        # "Pick any point on the elliptical slice."
        self.lecture[0].set_color(YELLOW)
        self.add(cone_left, cone_right, ellipse_line, sphere1, sphere2, f1, f2, f1_label, f2_label)
        self.play(FadeIn(p_dot), FadeIn(p_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Distance to focus equals distance to the tangent circle."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(pf1), Create(pq1))
        self.add(q1_dot, q1_label)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "Sum of distances equals the segment between tangent circles."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Create(pf2), Create(pq2))
        self.add(q2_dot, q2_label)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # "This segment length is constant along the cone's surface."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Segment along the cone generator
        q1q2_seg = Line(get_q_pos(t1_y), get_q_pos(t2_y), color=WHITE, stroke_width=6)
        q1q2_seg.add_updater(lambda l: l.put_start_and_end_on(get_q_pos(t1_y), get_q_pos(t2_y)))
        
        self.play(Create(q1q2_seg))
        self.wait(1)
        
        # Move P to show PF1+PF2 remains equivalent to Q1Q2
        self.play(p_val.animate.set_value(0.15), run_time=2, rate_func=linear)
        self.play(p_val.animate.set_value(0.85), run_time=3, rate_func=linear)
        self.play(p_val.animate.set_value(0.5), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        # "Thus, PF1 plus PF2 is always constant."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        formula = MathTex("PF_1 + PF_2 = PQ_1 + PQ_2 = Q_1 Q_2", font_size=26)
        # Resolved Issue 23:
        self.place_in_area(formula, 'F1', 'F6', scale_factor=0.75)
        self.play(Write(formula))
        
        self.play(p_val.animate.set_value(0.2), run_time=2)
        self.play(p_val.animate.set_value(0.7), run_time=2)
        
        self.wait(2)
