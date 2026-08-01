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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup initial layout
        lecture_lines = [
            'The matrix exponential e to At is a power series.',
            'Identity plus At plus higher-order matrix powers.',
            "Time 't' scales the evolution of the system.",
            'Matrix powers warp the space over time.',
            'This transforms initial states into future trajectories.'
        ]
        self.setup_layout("The Formal Definition of e^(At)", lecture_lines)

        # Matrix for evolution: A = [[0.2, -1.5], [1.5, 0.2]] (Spiral expansion)
        def get_matrix_at_t(t):
            exp_factor = np.exp(0.2 * t)
            angle = 1.5 * t
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return exp_factor * np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,      1]
            ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        # Replaced MathTex with Text to avoid dependence on 'latex' binary
        formula = Text("e^At = sum (At)^k / k!", color=WHITE, font_size=24)
        # FIXED: Issue 38 - Shifted to column 2 to avoid lecture notes
        self.place_in_area(formula, "A2", "A6", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        # Replaced MathTex with Text
        formula_expanded = Text("= I + At + (At)^2 / 2! + ...", color=WHITE, font_size=24)
        # FIXED: Issue 39 - Shifted to column 2 to maintain margin
        self.place_in_area(formula_expanded, "B2", "B6", scale_factor=0.8)
        self.play(FadeIn(formula_expanded, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        t_tracker = ValueTracker(0)
        
        # Slider UI
        slider_line = Line(self.grid["F2"], self.grid["F5"], color=WHITE)
        slider_dot = Dot(color=YELLOW)
        slider_dot.add_updater(lambda d: d.move_to(
            slider_line.point_from_proportion(t_tracker.get_value() / 2.0)
        ))
        # Replaced MathTex with Text
        t_label = Text("t = ", color=YELLOW, font_size=24)
        t_value = DecimalNumber(0, color=YELLOW, font_size=24, num_decimal_places=2, mob_class=Text)
        t_value.add_updater(lambda v: v.set_value(t_tracker.get_value()))
        
        self.place_at_grid(t_label, "F1", scale_factor=1.0)
        t_value.next_to(t_label, RIGHT, buff=0.1)
        
        self.add(slider_line, slider_dot, t_label, t_value)
        self.play(Create(slider_line), FadeIn(slider_dot))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        
        # Coordinate Plane
        plane = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.3}
        )
        # FIXED: Issue 40 - Shifted to column 2 to balance with left text
        self.place_in_area(plane, "C2", "E6", scale_factor=0.6)
        
        # Basis Vectors
        i_vec = Vector([1, 0], color=RED)
        j_vec = Vector([0, 1], color=GREEN)
        
        # Add Updaters for vectors
        def update_i(v):
            mat = get_matrix_at_t(t_tracker.get_value())
            new_end = mat @ np.array([1, 0, 0])
            v.put_start_and_end_on(plane.get_origin(), plane.c2p(new_end[0], new_end[1]))

        def update_j(v):
            mat = get_matrix_at_t(t_tracker.get_value())
            new_end = mat @ np.array([0, 1, 0])
            v.put_start_and_end_on(plane.get_origin(), plane.c2p(new_end[0], new_end[1]))

        i_vec.add_updater(update_i)
        j_vec.add_updater(update_j)
        
        # Trails
        i_trail = TracedPath(i_vec.get_end, stroke_color=RED, stroke_width=2)
        j_trail = TracedPath(j_vec.get_end, stroke_color=GREEN, stroke_width=2)
        
        self.add(plane, i_vec, j_vec, i_trail, j_trail)
        self.play(t_tracker.animate.set_value(2), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(BLUE)
        self.play(Indicate(self.lecture[4]))
        self.wait(2)
