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
        # LECTURE LINES
        lines = [
            "A linear combination scales and adds multiple vectors together.",
            "Vector-Bot uses these \"recipes\" to reach new coordinates.",
            "Watch how changing the scalars moves him across space.",
            "Scale vector v by a and vector w by b.",
            "Their sum a*v + b*w determines his final destination."
        ]
        
        self.setup_layout("Linear Combinations: The Recipe", lines)
        
        # Colors
        V_COLOR = "#FF6666" # Soft Red
        W_COLOR = "#66B2FF" # Soft Blue
        SUM_COLOR = "#FFFF66" # Soft Yellow
        
        # Assets
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        robot = SVGMobject(robot_path).set_color(WHITE)
        
        # === Animation for Lecture Line 1 ===
        # A linear combination scales and adds multiple vectors together.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Resolve Issue 22: Place formula in A2-A5
        formula = MathTex("a", "\\vec{v}", "+", "b", "\\vec{w}", color=WHITE)
        formula.set_color_by_tex("a", SUM_COLOR)
        formula.set_color_by_tex("b", SUM_COLOR)
        formula.set_color_by_tex("\\vec{v}", V_COLOR)
        formula.set_color_by_tex("\\vec{w}", W_COLOR)
        
        self.place_in_area(formula, "A2", "A5", scale_factor=1.2)
        
        # Resolve Issue 16: Use robot to introduce formula
        self.place_at_grid(robot, "A1", scale_factor=0.4)
        self.play(FadeIn(robot, shift=RIGHT))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vector-Bot uses these "recipes" to reach new coordinates.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Resolve Issue 21: Place plane in B1-F6
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=4, # Slightly taller to fill the B-F gap better
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        )
        self.place_in_area(plane, "B1", "F6")
        
        # Move robot to origin on plane
        self.play(
            robot.animate.scale(0.5).move_to(plane.c2p(0, 0)),
            Create(plane)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Watch how changing the scalars moves him across space.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # ValueTrackers for scalars
        a_tracker = ValueTracker(0)
        b_tracker = ValueTracker(0)
        
        # Robot movement updater
        # Note: SVG is lightweight enough for simple updater
        robot.add_updater(lambda r: r.move_to(plane.c2p(a_tracker.get_value(), b_tracker.get_value())))
        
        # Movement demonstration
        self.play(
            a_tracker.animate.set_value(2),
            b_tracker.animate.set_value(1),
            run_time=2
        )
        self.play(
            a_tracker.animate.set_value(-2),
            b_tracker.animate.set_value(-1),
            run_time=2
        )
        self.play(
            a_tracker.animate.set_value(0),
            b_tracker.animate.set_value(0),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Scale vector v by a and vector w by b.
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(WHITE)
        )
        
        # Base vectors
        v_base = Arrow(plane.c2p(0,0), plane.c2p(1,0), buff=0, color=V_COLOR)
        w_base = Arrow(plane.c2p(0,0), plane.c2p(0,1), buff=0, color=W_COLOR)
        
        v_label = MathTex("\\vec{v}", color=V_COLOR, font_size=20).next_to(v_base, DOWN, buff=0.1)
        w_label = MathTex("\\vec{w}", color=W_COLOR, font_size=20).next_to(w_base, LEFT, buff=0.1)

        self.play(GrowArrow(v_base), FadeIn(v_label))
        self.play(GrowArrow(w_base), FadeIn(w_label))
        
        # Target scale values: a=3, b=-2
        target_a = 3
        target_b = -2
        
        # Scaling vectors and moving robot
        self.play(
            v_base.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(target_a,0)),
            w_base.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(0,target_b)),
            a_tracker.animate.set_value(target_a),
            b_tracker.animate.set_value(target_b),
            v_label.animate.next_to(plane.c2p(target_a, 0), DOWN, buff=0.1),
            w_label.animate.next_to(plane.c2p(0, target_b), LEFT, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Their sum a*v + b*w determines his final destination.
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(SUM_COLOR)
        )
        
        # Visual vector addition (parallelogram/tip-to-tail)
        w_ghost = Arrow(plane.c2p(target_a, 0), plane.c2p(target_a, target_b), buff=0, color=W_COLOR, stroke_opacity=0.6)
        res_vector = Arrow(plane.c2p(0,0), plane.c2p(target_a, target_b), buff=0, color=SUM_COLOR)
        
        # Result label
        res_label = MathTex("3\\vec{v} - 2\\vec{w}", color=SUM_COLOR, font_size=24)
        # Position label near end of result vector - but use grid-relative logic if possible or next_to
        # Constraint B012: place label exactly one unit away (or use grid relative)
        # Since this is on a plane, next_to is safer for tracking.
        res_label.next_to(res_vector.get_end(), DR, buff=0.2)
        
        self.play(Create(w_ghost))
        self.play(GrowArrow(res_vector), FadeIn(res_label))
        
        self.wait(3)
