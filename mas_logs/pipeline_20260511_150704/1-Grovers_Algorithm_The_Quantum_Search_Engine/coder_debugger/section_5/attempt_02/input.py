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
        # Initial layout setup
        lecture_lines = [
            "Grover's algorithm performs a rotation in vector space.",
            "The state moves from uncertainty toward the target.",
            "Each iteration brings the vector closer to solution.",
            "We need roughly square root of N rotations.",
            "This geometric shift leads to the final measurement."
        ]
        self.setup_layout("Geometric Interpretation: The State Rotation", lecture_lines)

        # Colors
        COLOR_AXIS = WHITE
        COLOR_VECTOR = "#5271FF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_LECTURE_HIGHLIGHT = "#5271FF"

        # Parameters
        origin = self.grid["E2"]
        axis_len = 3.5
        theta_val = 12 * DEGREES # Initial angle from |s_perp>
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LECTURE_HIGHLIGHT)
        
        # Create Axes
        h_axis = Arrow(start=origin, end=origin + RIGHT * axis_len, color=COLOR_AXIS, buff=0)
        v_axis = Arrow(start=origin, end=origin + UP * axis_len, color=COLOR_AXIS, buff=0)
        
        # Fixed: Changed MathTex to Text to avoid LaTeX dependency error
        label_s_perp = Text("|s_⟂⟩", color=COLOR_AXIS, font_size=24)
        label_w = Text("|w⟩", color=COLOR_AXIS, font_size=24)
        
        self.place_at_grid(label_s_perp, "E6", scale_factor=1.0)
        self.place_at_grid(label_w, "A2", scale_factor=1.0)
        label_s_perp.shift(DOWN * 0.3)
        label_w.shift(LEFT * 0.5)

        self.play(Create(h_axis), Create(v_axis), Write(label_s_perp), Write(label_w))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LECTURE_HIGHLIGHT)
        
        # State vector
        state_vec_val = ValueTracker(theta_val)
        
        def get_vec_end():
            angle = state_vec_val.get_value()
            return origin + np.array([np.cos(angle), np.sin(angle), 0]) * axis_len * 0.8

        state_arrow = Arrow(start=origin, end=get_vec_end(), color=COLOR_VECTOR, buff=0)
        state_arrow.add_updater(lambda m: m.put_start_and_end_on(origin, get_vec_end()))
        
        # Fixed: Changed MathTex to Text
        label_psi = Text("|s⟩", color=COLOR_VECTOR, font_size=24)
        label_psi.add_updater(lambda m: m.move_to(get_vec_end() + RIGHT * 0.3 + UP * 0.2))

        self.play(Create(state_arrow), Write(label_psi))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_LECTURE_HIGHLIGHT)
        
        # First Grover Iteration: Rotate by 2*theta
        target_angle_1 = 3 * theta_val
        
        arc_1 = Arc(radius=1.0, start_angle=theta_val, angle=2*theta_val, arc_center=origin, color=WHITE)
        # Fixed: Changed MathTex to Text
        arc_label = Text("2θ", color=WHITE, font_size=20)
        arc_label.move_to(origin + np.array([np.cos(2*theta_val), np.sin(2*theta_val), 0]) * 1.3)

        self.play(state_vec_val.animate.set_value(target_angle_1), Create(arc_1), Write(arc_label), run_time=2)
        self.wait(1)
        self.play(FadeOut(arc_1), FadeOut(arc_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_LECTURE_HIGHLIGHT)
        
        # Subsequent rotations
        target_angle_2 = 5 * theta_val
        target_angle_final = 7 * theta_val
        
        self.play(state_vec_val.animate.set_value(target_angle_2), run_time=1)
        self.play(state_vec_val.animate.set_value(target_angle_final), run_time=1)
        
        total_arc = Arc(radius=0.7, start_angle=theta_val, angle=6*theta_val, arc_center=origin, color=WHITE)
        total_label = Text("~√N iterations", color=WHITE, font_size=18)
        self.place_at_grid(total_label, "D5", scale_factor=1.0)
        
        self.play(Create(total_arc), Write(total_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_LECTURE_HIGHLIGHT)
        
        state_arrow.clear_updaters()
        label_psi.clear_updaters()
        self.play(state_arrow.animate.set_color(COLOR_HIGHLIGHT), run_time=0.5)
        self.play(Flash(state_arrow.get_end(), color=COLOR_HIGHLIGHT, flash_radius=0.3))
        
        final_text = Text("Found!", color=COLOR_HIGHLIGHT, font_size=22)
        self.place_at_grid(final_text, "B3", scale_factor=1.0)
        self.play(Write(final_text))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)