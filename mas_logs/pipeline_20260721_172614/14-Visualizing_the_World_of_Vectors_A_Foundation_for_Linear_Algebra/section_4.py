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
        # 1. Setup layout
        title_text = "Scalar Multiplication: Scaling the Magnitude"
        lecture_lines = [
            "Multiplying a vector by a scalar changes its length.",
            "Positive scalars stretch or shrink the arrow’s size.",
            "Negative scalars flip the direction to the opposite side."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        VEC_COLOR = "#00FA9A"
        SCALAR_COLOR = "#FFD700"
        NEG_COLOR = "#FF4500"

        # Coordinate Plane - simplified to ensure performance
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, 'A1', 'F6')
        
        # === Animation for Lecture Line 1 ===
        # Color change for current line
        self.lecture[0].set_color(VEC_COLOR)
        
        vec_start = plane.c2p(0, 0, 0)
        vec_end = plane.c2p(2, 1, 0)
        vector = Arrow(vec_start, vec_end, buff=0, color=VEC_COLOR, stroke_width=5)
        v_label = MathTex(r"\vec{v}", color=VEC_COLOR, font_size=30)
        v_label.move_to(vec_end + UR * 0.2)
        
        self.play(Create(plane), FadeIn(vector), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SCALAR_COLOR)
        
        # Scalar indicator
        scalar_val = ValueTracker(1.0)
        scalar_label = MathTex("c = ", color=SCALAR_COLOR, font_size=32)
        scalar_num = DecimalNumber(1.0, num_decimal_places=1, color=SCALAR_COLOR, font_size=32)
        scalar_grp = VGroup(scalar_label, scalar_num).arrange(RIGHT, buff=0.1)
        
        # Fix for Issue 24: Position at A1, scale 1.2
        self.place_at_grid(scalar_grp, 'A1', scale_factor=1.2)
        
        # Use updater for decimal number only
        scalar_num.add_updater(lambda d: d.set_value(scalar_val.get_value()))
        
        self.play(FadeIn(scalar_grp))
        
        # Stretch to 2.0
        new_end_2 = plane.c2p(4, 2, 0)
        self.play(
            vector.animate.put_start_and_end_on(vec_start, new_end_2),
            v_label.animate.move_to(new_end_2 + UR * 0.2),
            scalar_val.animate.set_value(2.0),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Shrink to 0.5
        new_end_05 = plane.c2p(1, 0.5, 0)
        self.play(
            vector.animate.put_start_and_end_on(vec_start, new_end_05),
            v_label.animate.move_to(new_end_05 + UR * 0.2),
            scalar_val.animate.set_value(0.5),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(NEG_COLOR)
        
        # Flip to -1.0
        new_end_neg1 = plane.c2p(-2, -1, 0)
        self.play(
            vector.animate.put_start_and_end_on(vec_start, new_end_neg1).set_color(NEG_COLOR),
            v_label.animate.move_to(new_end_neg1 + DL * 0.2).set_color(NEG_COLOR),
            scalar_val.animate.set_value(-1.0),
            scalar_grp.animate.set_color(NEG_COLOR),
            run_time=1.5
        )
        self.wait(1.5)
        
        scalar_num.clear_updaters()
