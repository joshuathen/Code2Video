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
            "Summing harmonics builds complex waves piece by piece.",
            "Adding more terms reveals sharp square wave edges.",
            "Gibbs phenomenon appears at the sharp corners."
        ]
        self.setup_layout("Step-by-Step Construction: Synthesis", lecture_lines)
        self.lecture.set_opacity(0)
        
        axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        # Applying layout fix from Issue 35 (most reasonable/centered/balanced recommendation)
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.6)
        self.add(axes)

        # Asset loading
        sine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg")
        square_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0].set_color("#FFFFFF")))
        base_line = Line(axes.c2p(-3, 0), axes.c2p(3, 0), color="#FFFFFF")
        self.place_at_grid(sine_icon, 'B4', scale_factor=0.5)
        self.play(Create(base_line), FadeIn(sine_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1].set_color("#FFFF00")))
        
        fund = axes.plot(lambda x: np.sin(x * np.pi), color="#FFFF00")
        self.play(Create(fund))
        
        # Second harmonic
        self.play(FadeIn(self.lecture[2].set_color("#00FFFF")))
        harm2 = axes.plot(lambda x: np.sin(x * np.pi) + (1/3)*np.sin(3 * x * np.pi), color="#00FFFF")
        self.play(Transform(fund, harm2))
        
        # Final convergence representation with square asset
        final_sum = axes.plot(lambda x: sum((4/np.pi) * (np.sin((2*n-1)*x*np.pi)/(2*n-1)) for n in range(1, 10)), color="#00FF00")
        self.place_at_grid(square_icon, 'B5', scale_factor=0.5)
        self.play(Transform(fund, final_sum), FadeIn(square_icon), run_time=2)
        
        self.wait(2)
