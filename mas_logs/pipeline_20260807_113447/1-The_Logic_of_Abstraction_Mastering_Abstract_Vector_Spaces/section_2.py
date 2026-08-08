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
        # 1. Setup layout
        self.setup_layout(
            "Prerequisite Check: The Two Pillars",
            [
                "Two fundamental operations support everything.",
                "Addition combines vectors tip-to-tail.",
                "Scalar multiplication scales their magnitude."
            ]
        )
        
        # Colors
        COLOR_U = RED
        COLOR_V = BLUE
        COLOR_SUM = YELLOW
        COLOR_W = PURPLE
        COLOR_SCALE = ORANGE
        
        # Dim lecture items initially
        for line in self.lecture:
            line.set_color(GRAY_E)

        # --- Step 1: Fundamentals ---
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Show a coordinate system in the visual area
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "stroke_width": 2, "color": GREY_B},
            tips=True
        ).move_to(self.get_pos("C3") + RIGHT * 0.5)
        
        self.play(Create(axes))
        self.wait(1)

        # --- Step 2: Addition ---
        self.play(
            self.lecture[0].animate.set_color(GRAY_D),
            self.lecture[1].animate.set_color(COLOR_SUM)
        )
        
        u_start = self.get_pos("E2")
        u_end = self.get_pos("C3")
        v_end = self.get_pos("B5")
        
        u_vec = Arrow(u_start, u_end, buff=0, color=COLOR_U, stroke_width=4)
        v_vec = Arrow(u_end, v_end, buff=0, color=COLOR_V, stroke_width=4)
        sum_vec = Arrow(u_start, v_end, buff=0, color=COLOR_SUM, stroke_width=6)
        
        u_label = MathTex(r"\vec{u}", color=COLOR_U).scale(0.8).next_to(u_vec, LEFT, buff=0.1)
        v_label = MathTex(r"\vec{v}", color=COLOR_V).scale(0.8).next_to(v_vec, UP, buff=0.1)
        sum_label = MathTex(r"\vec{u}+\vec{v}", color=COLOR_SUM).scale(0.8).next_to(sum_vec, DOWN, buff=0.2)

        self.play(GrowArrow(u_vec), Write(u_label))
        self.play(GrowArrow(v_vec), Write(v_label))
        self.wait(0.5)
        self.play(GrowArrow(sum_vec), Write(sum_label))
        self.wait(2)

        # --- Step 3: Scalar Multiplication ---
        self.play(
            self.lecture[1].animate.set_color(GRAY_D),
            self.lecture[2].animate.set_color(COLOR_SCALE),
            FadeOut(u_vec, v_vec, sum_vec, u_label, v_label, sum_label)
        )
        
        w_start = self.get_pos("D2")
        w_unit_end = self.get_pos("D3")
        
        # Initial vector w
        w_vec = Arrow(w_start, w_unit_end, buff=0, color=COLOR_W, stroke_width=5)
        w_label = MathTex(r"\vec{w}", color=COLOR_W).scale(0.9).next_to(w_start, LEFT, buff=0.2)
        
        self.play(GrowArrow(w_vec), Write(w_label))
        self.wait(1)
        
        # Scalar tracker for stretch
        scale_tracker = ValueTracker(1.0)
        
        # Dynamic updater for the arrow
        w_vec.add_updater(
            lambda m: m.put_start_and_end_on(
                w_start, 
                w_start + scale_tracker.get_value() * (w_unit_end - w_start)
            )
        )
        
        # Label that changes
        scaled_label = MathTex(r"3\vec{w}", color=COLOR_SCALE).scale(0.9).next_to(w_start, LEFT, buff=0.2)
        
        self.play(
            scale_tracker.animate.set_value(3.0),
            w_vec.animate.set_color(COLOR_SCALE),
            Transform(w_label, scaled_label),
            run_time=2,
            rate_func=smooth
        )
        
        w_vec.clear_updaters()
        self.wait(2)
