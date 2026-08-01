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
        # Colors
        COLOR_S = "#3498db"
        COLOR_I = "#e74c3c"
        COLOR_R = "#2ecc71"
        COLOR_BETA = "#f1c40f"
        COLOR_GAMMA = "#9b59b6"

        lecture_lines = [
            'Differential equations describe population shifts mathematically.',
            'The Susceptible group shrinks as people get sick.',
            'The Infected group grows, peaks, and then falls.',
            'Recovered numbers rise as patients heal.',
            "These equations predict the epidemic's trajectory."
        ]

        self.setup_layout("The SIR Equations Visualized", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Equation 1: dS/dt = -βSI/N
        self.lecture[0].set_color(YELLOW)
        
        eq_s = VGroup(
            Text("d", font_size=24), Text("S", font_size=24, color=COLOR_S), Text("/dt = -", font_size=24),
            Text("β", font_size=24, color=COLOR_BETA), Text("S", font_size=24, color=COLOR_S),
            Text("I", font_size=24, color=COLOR_I), Text("/N", font_size=24)
        ).arrange(RIGHT, buff=0.1)
        
        # Issue 47 Fix: reduce scale factor to 0.7
        self.place_in_area(eq_s, 'A1', 'A2', scale_factor=0.7)
        self.play(FadeIn(eq_s))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Equation 2: dI/dt = βSI/N - γI
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        eq_i = VGroup(
            Text("d", font_size=24), Text("I", font_size=24, color=COLOR_I), Text("/dt = ", font_size=24),
            Text("β", font_size=24, color=COLOR_BETA), Text("S", font_size=24, color=COLOR_S),
            Text("I", font_size=24, color=COLOR_I), Text("/N - ", font_size=24),
            Text("γ", font_size=24, color=COLOR_GAMMA), Text("I", font_size=24, color=COLOR_I)
        ).arrange(RIGHT, buff=0.1)
        
        # Issue 48 Fix: reduce scale factor to 0.7
        self.place_in_area(eq_i, 'A3', 'A4', scale_factor=0.7)
        self.play(FadeIn(eq_i))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Equation 3: dR/dt = γI
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        eq_r = VGroup(
            Text("d", font_size=24), Text("R", font_size=24, color=COLOR_R), Text("/dt = ", font_size=24),
            Text("γ", font_size=24, color=COLOR_GAMMA), Text("I", font_size=24, color=COLOR_I)
        ).arrange(RIGHT, buff=0.1)
        
        # Issue 49 Fix: reduce scale factor to 0.7
        self.place_in_area(eq_r, 'A5', 'A6', scale_factor=0.7)
        self.play(FadeIn(eq_r))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Coordinate Graph
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1.2, 0.2],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=5,
            y_length=3
        )
        x_label = Text("Time", font_size=18).next_to(axes.x_axis, DOWN)
        y_label = Text("Population", font_size=18).rotate(90*DEGREES).next_to(axes.y_axis, LEFT)
        graph_group = VGroup(axes, x_label, y_label)
        
        self.place_in_area(graph_group, 'B1', 'F6', scale_factor=0.8)
        self.play(FadeIn(graph_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Plotting the curves
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Approximate SIR curves using logistic and bell functions
        s_curve = axes.plot(lambda t: 1.0 / (1 + np.exp(t - 5)), color=COLOR_S)
        i_curve = axes.plot(lambda t: 0.8 * np.exp(-0.4 * (t - 5)**2), color=COLOR_I)
        r_curve = axes.plot(lambda t: 1.0 - (1.0 / (1 + np.exp(t - 5))), color=COLOR_R)
        
        self.play(
            Create(s_curve),
            Create(i_curve),
            Create(r_curve),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
