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
        # Setup context with precise script lines
        lines = [
            "Superposition is a linear combination of these base states.",
            "Every valid state vector lies on a unit circle.",
            "Rotating the vector changes the weight of each state.",
            "Dashed lines represent probability amplitudes, alpha and beta.",
            "The squared magnitudes must always sum to one."
        ]
        self.setup_layout("Defining Superposition Math", lines)
        
        # Colors
        CYAN = "#00FFFF"
        GREY_DARK = "#555555"
        GREY_LIGHT = "#888888"
        YELLOW = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Linear combination formula: |ψ⟩ = α|0⟩ + β|1⟩
        f1_1 = Text("|ψ⟩ = ", font_size=32)
        f1_2 = Text("α", font_size=32, color=CYAN)
        f1_3 = Text("|0⟩ + ", font_size=32)
        f1_4 = Text("β", font_size=32, color=CYAN)
        f1_5 = Text("|1⟩", font_size=32)
        formula1 = VGroup(f1_1, f1_2, f1_3, f1_4, f1_5).arrange(RIGHT, buff=0.1)
        # Fix Issue 40: Adjust area to remove vertical gap
        self.place_in_area(formula1, "A2", "B5", scale_factor=0.9)

        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(formula1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Unit circle and initial vector
        circle = Circle(radius=1.5, color=GREY_DARK)
        h_axis = Line(LEFT * 1.7, RIGHT * 1.7, color=GREY_DARK)
        v_axis = Line(DOWN * 1.7, UP * 1.7, color=GREY_DARK)
        axes_group = VGroup(circle, h_axis, v_axis)
        # Fix Issue 38: Expand horizontal area to E6 and scale down slightly
        self.place_in_area(axes_group, "C2", "E6", scale_factor=0.9)
        center = circle.get_center()
        radius = 1.5

        # Basis labels
        zero_label = Text("|0⟩", font_size=20).move_to(center + RIGHT * 1.8)
        one_label = Text("|1⟩", font_size=20).move_to(center + UP * 1.8)

        angle_tracker = ValueTracker(PI/6) 
        
        vector_psi = Arrow(stroke_width=4, color=WHITE, buff=0)
        vector_psi.add_updater(lambda m: m.put_start_and_end_on(
            center, 
            center + np.array([radius * np.cos(angle_tracker.get_value()), radius * np.sin(angle_tracker.get_value()), 0])
        ))
        
        psi_label = Text("|ψ⟩", font_size=22)
        psi_label.add_updater(lambda m: m.move_to(
            center + 1.2 * np.array([radius * np.cos(angle_tracker.get_value()), radius * np.sin(angle_tracker.get_value()), 0])
        ))

        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            FadeIn(axes_group),
            FadeIn(zero_label),
            FadeIn(one_label),
            FadeIn(vector_psi),
            FadeIn(psi_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Set up dynamic components for rotation
        line_x = DashedLine(color=GREY_LIGHT)
        line_x.add_updater(lambda m: m.put_start_and_end_on(
            center + np.array([radius * np.cos(angle_tracker.get_value()), 0, 0]),
            center + np.array([radius * np.cos(angle_tracker.get_value()), radius * np.sin(angle_tracker.get_value()), 0])
        ))
        line_y = DashedLine(color=GREY_LIGHT)
        line_y.add_updater(lambda m: m.put_start_and_end_on(
            center + np.array([0, radius * np.sin(angle_tracker.get_value()), 0]),
            center + np.array([radius * np.cos(angle_tracker.get_value()), radius * np.sin(angle_tracker.get_value()), 0])
        ))
        
        label_alpha = Text("α", font_size=24, color=CYAN)
        label_alpha.add_updater(lambda m: m.move_to(
            center + np.array([radius * np.cos(angle_tracker.get_value()) / 2, -0.3, 0])
        ))
        label_beta = Text("β", font_size=24, color=CYAN)
        label_beta.add_updater(lambda m: m.move_to(
            center + np.array([-0.3, radius * np.sin(angle_tracker.get_value()) / 2, 0])
        ))

        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Rotation animation with dashed lines updating
        self.play(
            Create(line_x), Create(line_y),
            FadeIn(label_alpha), FadeIn(label_beta)
        )
        self.play(
            angle_tracker.animate.set_value(PI/3),
            run_time=2,
            rate_func=linear
        )
        self.play(
            angle_tracker.animate.set_value(PI/12),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Focus on probability amplitudes
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(
            Indicate(label_alpha),
            Indicate(label_beta),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Normalization equation: |α|² + |β|² = 1
        n_alpha = Text("|α|²", font_size=32, color=YELLOW)
        n_plus = Text(" + ", font_size=32, color=YELLOW)
        n_beta = Text("|β|²", font_size=32, color=YELLOW)
        n_eq = Text(" = 1", font_size=32, color=YELLOW)
        formula2 = VGroup(n_alpha, n_plus, n_beta, n_eq).arrange(RIGHT, buff=0.1)
        # Fix Issue 39: Place at F2-F5 with scale 0.8 to avoid edge
        self.place_in_area(formula2, "F2", "F5", scale_factor=0.8)

        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(FadeIn(formula2))
        self.wait(2)
