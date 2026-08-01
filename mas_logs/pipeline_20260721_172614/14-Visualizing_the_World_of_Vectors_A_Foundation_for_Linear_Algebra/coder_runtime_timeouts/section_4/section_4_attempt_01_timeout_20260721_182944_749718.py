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

        # Define colors for elements
        VEC_COLOR = "#00FA9A"      # Medium Spring Green
        SCALAR_COLOR = "#FFD700"   # Gold
        NEGATIVE_COLOR = "#FF4500" # OrangeRed

        # Prepare background plane in the right-side area
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5.5,
            y_length=5.5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'A1', 'F6')

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(VEC_COLOR)
        
        # Initial vector v = [2, 1]
        v_coords = np.array([2, 1, 0])
        vector = Arrow(
            plane.c2p(0, 0, 0),
            plane.c2p(v_coords[0], v_coords[1], 0),
            buff=0,
            color=VEC_COLOR,
            stroke_width=6
        )
        v_label = MathTex(r"\vec{v}", color=VEC_COLOR, font_size=32)
        v_label.next_to(vector.get_end(), UR, buff=0.1)

        self.play(Create(plane), FadeIn(vector), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlight
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SCALAR_COLOR)
        
        # Scalar tracker and UI elements
        scalar_val = ValueTracker(1.0)
        scalar_label = MathTex("c = ", color=SCALAR_COLOR, font_size=36)
        scalar_num = DecimalNumber(1.0, num_decimal_places=1, color=SCALAR_COLOR, font_size=36)
        scalar_display = VGroup(scalar_label, scalar_num).arrange(RIGHT, buff=0.1)
        self.place_at_grid(scalar_display, 'A2')

        # Define updaters for smooth transformation
        def update_vec(v):
            val = scalar_val.get_value()
            new_end = plane.c2p(2 * val, 1 * val, 0)
            v.put_start_and_end_on(plane.c2p(0, 0, 0), new_end)
            # Dynamic color change for negative scalar
            if val < 0:
                v.set_color(NEGATIVE_COLOR)
            else:
                v.set_color(VEC_COLOR)

        def update_v_label(l):
            val = scalar_val.get_value()
            end_pos = vector.get_end()
            # Adjust label position and color based on vector direction
            if val >= 0:
                l.next_to(end_pos, UR, buff=0.1)
                l.set_color(VEC_COLOR)
            else:
                l.next_to(end_pos, DL, buff=0.1)
                l.set_color(NEGATIVE_COLOR)

        # Link mobjects to the ValueTracker
        scalar_num.add_updater(lambda d: d.set_value(scalar_val.get_value()))
        vector.add_updater(update_vec)
        v_label.add_updater(update_v_label)

        self.play(FadeIn(scalar_display))
        self.wait(0.5)
        
        # Stretch vector: c=1.0 -> c=2.0
        self.play(scalar_val.animate.set_value(2.0), run_time=2, rate_func=linear)
        self.wait(1)
        
        # Shrink vector: c=2.0 -> c=0.5
        self.play(scalar_val.animate.set_value(0.5), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlight
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(NEGATIVE_COLOR)
        
        # Transition scalar display color to match the negative state
        self.play(
            scalar_label.animate.set_color(NEGATIVE_COLOR),
            scalar_num.animate.set_color(NEGATIVE_COLOR)
        )
        
        # Flip vector: c=0.5 -> c=-1.0
        self.play(scalar_val.animate.set_value(-1.0), run_time=2, rate_func=linear)
        self.wait(2)
        
        # Cleanup updaters
        vector.clear_updaters()
        v_label.clear_updaters()
        scalar_num.clear_updaters()
