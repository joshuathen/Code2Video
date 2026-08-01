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
        # Initialize basic layout
        lines = [
            "Beta measures the likelihood of disease transmission.",
            "Higher Beta values speed up the infection's spread.",
            "Gamma controls the rate at which people recover."
        ]
        self.setup_layout("The Dynamics: Beta (β) and Gamma (γ)", lines)

        # Colors
        BETA_COLOR = "#f1c40f"  # Yellow
        GAMMA_COLOR = "#9b59b6" # Purple
        S_COLOR = "#3498db"
        I_COLOR = "#e74c3c"
        R_COLOR = "#2ecc71"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BETA_COLOR)
        
        # S and I compartments
        s_box = RoundedRectangle(corner_radius=0.1, height=1.2, width=1.2, color=S_COLOR)
        s_label = Text("S", color=S_COLOR)
        s_group = VGroup(s_box, s_label)
        self.place_in_area(s_group, "A2", "B2", scale_factor=0.8)

        i_box = RoundedRectangle(corner_radius=0.1, height=1.2, width=1.2, color=I_COLOR)
        i_label = Text("I", color=I_COLOR)
        i_group = VGroup(i_box, i_label)
        self.place_in_area(i_group, "A4", "B4", scale_factor=0.8)

        # Beta Arrow (S to I)
        beta_arrow = Arrow(
            start=s_group.get_right(),
            end=i_group.get_left(),
            buff=0.1,
            color=BETA_COLOR,
            stroke_width=2
        )
        beta_sym = Text("β", color=BETA_COLOR, font_size=32, weight=BOLD)
        self.place_at_grid(beta_sym, "A3", scale_factor=1.0)
        
        self.play(Create(s_group), Create(i_group))
        self.play(GrowArrow(beta_arrow), FadeIn(beta_sym))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BETA_COLOR)

        # Beta Slider
        beta_slider_track = Line(self.grid["C2"], self.grid["C4"], color=GRAY)
        beta_slider_knob = Dot(self.grid["C2"], color=BETA_COLOR)
        beta_slider_label = Text("Beta β", font_size=20, color=BETA_COLOR)
        self.place_at_grid(beta_slider_label, "C1", scale_factor=1.0)
        
        self.play(Create(beta_slider_track), FadeIn(beta_slider_knob), FadeIn(beta_slider_label))
        
        # Increase Beta Visuals - Slider move and Arrow thickens
        self.play(
            beta_slider_knob.animate.move_to(self.grid["C4"]),
            beta_arrow.animate.set_stroke(width=12),
            run_time=1.5
        )
        
        # Moving dots to show rapid spread from S to I
        dot_path = Line(s_group.get_right(), i_group.get_left())
        dots = VGroup(*[Dot(radius=0.08, color=BETA_COLOR) for _ in range(8)])
        
        self.play(
            LaggedStart(
                *[MoveAlongPath(dot, dot_path) for dot in dots],
                lag_ratio=0.1,
                run_time=1.5
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GAMMA_COLOR)

        # R compartment
        r_box = RoundedRectangle(corner_radius=0.1, height=1.2, width=1.2, color=R_COLOR)
        r_label = Text("R", color=R_COLOR)
        r_group = VGroup(r_box, r_label)
        self.place_in_area(r_group, "A6", "B6", scale_factor=0.8)

        # Gamma Arrow (I to R)
        gamma_arrow = Arrow(
            start=i_group.get_right(),
            end=r_group.get_left(),
            buff=0.1,
            color=GAMMA_COLOR,
            stroke_width=4
        )
        gamma_sym = Text("γ", color=GAMMA_COLOR, font_size=32)
        self.place_at_grid(gamma_sym, "A5", scale_factor=1.0)

        # Gamma Slider for interaction consistency
        gamma_slider_track = Line(self.grid["E2"], self.grid["E4"], color=GRAY)
        gamma_slider_knob = Dot(self.grid["E2"], color=GAMMA_COLOR)
        gamma_slider_label = Text("Gamma γ", font_size=20, color=GAMMA_COLOR)
        # Resolved Issue 46: Label placement at E1 to align with Beta Slider label at C1
        self.place_at_grid(gamma_slider_label, "E1", scale_factor=1.0)

        self.play(Create(r_group), GrowArrow(gamma_arrow), FadeIn(gamma_sym))
        self.play(Create(gamma_slider_track), FadeIn(gamma_slider_knob), FadeIn(gamma_slider_label))
        
        # Interaction: Gamma increases
        self.play(
            gamma_slider_knob.animate.move_to(self.grid["E4"]),
            gamma_arrow.animate.set_stroke(width=12),
            run_time=1.5
        )
        
        # Final flow dots through the whole system: S -> I -> R
        path_si = Line(s_group.get_right(), i_group.get_left())
        path_ir = Line(i_group.get_right(), r_group.get_left())
        
        def get_flow_animations():
            flow_dots = VGroup(*[Dot(radius=0.08, color=WHITE) for _ in range(10)])
            anims = []
            for dot in flow_dots:
                anims.append(Succession(
                    MoveAlongPath(dot, path_si, run_time=0.6),
                    MoveAlongPath(dot, path_ir, run_time=0.6)
                ))
            return anims

        self.play(
            LaggedStart(*get_flow_animations(), lag_ratio=0.3),
            run_time=4
        )
        self.wait(2)
