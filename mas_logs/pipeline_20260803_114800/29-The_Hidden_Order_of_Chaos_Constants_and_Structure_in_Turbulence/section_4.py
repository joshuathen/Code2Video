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
        lecture_lines = [
            "Energy distribution follows a strict mathematical power law.",
            "In the inertial subrange, energy scales with size.",
            "The spectrum E(k) relates energy to the wavenumber k.",
            "Energy drops precisely along a negative 5/3 slope.",
            "This -5/3 law is the fingerprint of turbulence."
        ]
        
        self.setup_layout("The Kolmogorov 5/3 Law", lecture_lines)
        
        # Define Colors
        GOLD = "#FFD700"
        CYAN = "#00FFFF"
        TEAL = "#008080"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display the formula E(k) = C * eps^(2/3) * k^(-5/3); highlight k^(-5/3) in gold (#FFD700).
        formula = MathTex(
            "E(k)", "=", "C", "\\epsilon^{2/3}", "k^{-5/3}",
            font_size=36
        )
        # Resolved Issue 32: Adjusted position
        self.place_in_area(formula, 'A3', 'B6')
        
        self.play(
            self.lecture[0].animate.set_color(GOLD),
            Write(formula)
        )
        self.play(formula[4].animate.set_color(GOLD))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a Log-Log graph (Energy vs Wavenumber) with a straight line having a -5/3 slope.
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "include_ticks": False},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        )
        x_label = MathTex("\\log(k)", font_size=20).next_to(axes.x_axis, DOWN, buff=0.1)
        y_label = MathTex("\\log(E)", font_size=20).rotate(PI/2).next_to(axes.y_axis, LEFT, buff=0.1)
        graph_group = VGroup(axes, x_label, y_label)
        # Resolved Issue 33: Adjusted position
        self.place_in_area(graph_group, 'C3', 'F6')
        
        # Linear line on log-log is log(E) = -5/3 log(k) + C
        slope_line = Line(
            axes.c2p(1, 8), axes.c2p(8, 1),
            color=CYAN, stroke_width=4
        )

        self.play(
            self.lecture[1].animate.set_color(CYAN),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        self.play(Create(slope_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Label the slope "-5/3" in gold (#FFD700) and highlight the line.
        slope_label = MathTex("\\text{Slope} = -5/3", font_size=22, color=GOLD)
        slope_label.next_to(slope_line, UR, buff=-0.3)
        
        self.play(
            self.lecture[2].animate.set_color(GOLD),
            Write(slope_label),
            slope_line.animate.set_color(GOLD).set_stroke(width=6)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition graph to "Graphic Equalizer" bars [Asset: ...]; high on the left, dropping quickly to the right.
        # Resolved Issue 21: Asset integration
        equalizer_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/equalizer.svg")
        self.place_in_area(equalizer_asset, 'C3', 'F6', scale_factor=0.8)
        equalizer_asset.set_color(TEAL)

        self.play(
            self.lecture[3].animate.set_color(TEAL),
            FadeOut(slope_line),
            FadeOut(slope_label),
            FadeOut(graph_group),
            FadeIn(equalizer_asset)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the "Inertial Subrange" region on the graph in white (#FFFFFF).
        subrange_rect = Rectangle(
            width=2.5, height=2,
            stroke_color=WHITE_COLOR,
            fill_color=WHITE_COLOR,
            fill_opacity=0.2
        )
        # Resolved Issue 34: Adjusted position and scale
        self.place_in_area(subrange_rect, 'D4', 'E5', scale_factor=1.3)
        subrange_label = Text("Inertial Subrange", font_size=18, color=WHITE_COLOR)
        subrange_label.next_to(subrange_rect, UP, buff=0.1)

        self.play(
            self.lecture[4].animate.set_color(WHITE_COLOR),
            Create(subrange_rect),
            Write(subrange_label)
        )
        self.wait(2)
