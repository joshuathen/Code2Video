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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Flow transitions from smooth laminar to chaotic turbulence.",
            "This shift depends on the dimensionless Reynolds Number.",
            "Viscosity acts as internal friction, resisting this change."
        ]
        self.setup_layout("The Transition to Chaos", lecture_lines)

        # Colors
        COLOR_LAMINAR = "#0000FF"
        COLOR_LAMINAR_LABEL = "#ADD8E6"
        COLOR_CRITICAL = "#FF0000"
        HIGHLIGHT_COLOR = YELLOW

        # --- Assets/Mobjects ---
        # Shark Asset (Resolving Issue 19)
        shark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shark.svg")
        shark.set_color(GREY_B)
        self.place_in_area(shark, 'B2', 'C3', scale_factor=0.6)

        # Laminar lines
        num_lines = 4
        stream_lines = VGroup()
        for i in range(num_lines):
            row_char = ["B", "C", "D", "E"][i]
            start_pt = self.grid[f"{row_char}1"]
            end_pt = self.grid[f"{row_char}6"]
            l = Line(start_pt, end_pt, color=COLOR_LAMINAR, stroke_width=2)
            stream_lines.add(l)

        laminar_label = Text("Laminar Flow", font_size=24, color=COLOR_LAMINAR_LABEL)
        # Issue 23 Fix: place_in_area('A3', 'A4')
        self.place_in_area(laminar_label, "A3", "A4")

        # Eddies (for turbulence)
        eddies = VGroup()
        for pos in ["C4", "C5", "D4", "D5"]:
            swirl = ParametricFunction(
                lambda t: np.array([0.15*t*np.cos(8*t), 0.15*t*np.sin(8*t), 0]),
                t_range=[0, 1.5], color=COLOR_LAMINAR
            )
            self.place_at_grid(swirl, pos)
            eddies.add(swirl)

        # Reynolds Number / Critical Text
        critical_text = Text("Critical Re Threshold", font_size=24, color=COLOR_CRITICAL)
        # Issue 25 Fix: place_in_area('F4', 'F6')
        self.place_in_area(critical_text, "F4", "F6")

        # Viscosity label
        visc_label = Text("Viscosity (Friction)", font_size=20, color=WHITE)
        # Issue 24 Fix: place_in_area('F1', 'F2')
        self.place_in_area(visc_label, "F1", "F2")

        # === Animation for Lecture Line 1 ===
        # Apply ONLY color changes to lecture lines
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(FadeIn(shark), Create(stream_lines), Write(laminar_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Wobbly lines for transition
        wobble_lines = VGroup()
        for i in range(num_lines):
            row_char = ["B", "C", "D", "E"][i]
            y_base = self.grid[f"{row_char}1"][1]
            x_start = self.grid[f"{row_char}1"][0]
            x_end = self.grid[f"{row_char}6"][0]
            sin_line = FunctionGraph(
                lambda x, y_b=y_base, x_s=x_start: y_b + 0.15 * np.sin(4 * (x - x_s)),
                x_range=[x_start, x_end],
                color=COLOR_LAMINAR
            )
            wobble_lines.add(sin_line)

        # Animate shark acceleration and flow transition
        self.play(
            ReplacementTransform(stream_lines, wobble_lines),
            FadeIn(eddies, scale=0.5),
            shark.animate.shift(RIGHT * 0.4),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(Write(critical_text))
        
        friction_arrows = VGroup(*[
            Arrow(start=self.grid[f"E{i}"], end=self.grid[f"E{i}"] + LEFT*0.4, color=WHITE, buff=0, stroke_width=2)
            for i in range(2, 6)
        ])
        
        self.play(Create(friction_arrows), FadeIn(visc_label))
        self.wait(2)

        # Reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
