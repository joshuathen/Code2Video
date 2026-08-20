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
            "Adding terms creates sharper, more accurate waves.",
            "Convergence improves as we add more terms.",
            "Gibbs phenomenon causes overshoot at signal corners."
        ]
        self.setup_layout("Visualizing Convergence: The Square Wave Demo", lecture_lines)
        
        # Asset: oscilloscope icon
        # Load asset once
        oscilloscope_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscilloscope.svg"
        
        # Axes for visualization - applied suggested layout change (Line 57)
        axes = Axes(x_range=[-PI, PI], y_range=[-1.5, 1.5], axis_config={"include_tip": False})
        # Applying layout fix from issue 46
        self.place_in_area(axes, "A3", "F6", scale_factor=0.4)
        
        # Target Square Wave
        square_wave = axes.plot(lambda x: 1 if x > 0 else -1, color=WHITE)
        
        # Create asset wrapper
        # We place the oscilloscope icon as a background or frame for the graph
        scope_icon = SVGMobject(oscilloscope_path)
        self.place_at_grid(scope_icon, "C3", scale_factor=0.5)

        # Initial state: Show square wave outline
        self.add(square_wave)
        
        # Function to generate Fourier sum
        def fourier_sum(n_terms):
            def func(x):
                res = 0
                for i in range(n_terms):
                    n = 2 * i + 1
                    res += (4 / (np.pi * n)) * np.sin(n * x)
                return res
            return func

        # === Animation for Lecture Line 1 ===
        # Adding terms creates sharper, more accurate waves.
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        f1 = axes.plot(fourier_sum(1), color="#FF00FF")
        self.play(Create(f1))

        # === Animation for Lecture Line 2 ===
        # Convergence improves as we add more terms.
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        f10 = axes.plot(fourier_sum(10), color="#00FFFF")
        self.play(Transform(f1, f10))
        
        # Error bar
        error_bar = Line(start=axes.c2p(2, 0), end=axes.c2p(2, 0.5), color="#FFFF00")
        self.play(Create(error_bar))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Gibbs phenomenon causes overshoot at signal corners.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Show overshoot zoom/highlight
        overshoot = Circle(radius=0.2, color=RED).move_to(axes.c2p(0, 1.2))
        self.play(Create(overshoot))
        self.wait(2)
