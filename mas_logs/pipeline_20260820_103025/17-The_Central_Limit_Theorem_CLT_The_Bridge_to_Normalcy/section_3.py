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
        lecture_lines = [
            "Populations can be any random shape.",
            "Sample means form a normal distribution.",
            "The bell curve stabilizes with larger samples.",
            "The center matches the population mean.",
            "It's the CLT bridging chaos to order."
        ]
        self.setup_layout("The Big Reveal: The Normal Distribution", lecture_lines)
        
        # Define the bell curve
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#4287f5", stroke_width=4)
        peak_line = Line(start=axes.c2p(0, 0), end=axes.c2p(0, 0.4), color="#FFFF00", stroke_width=4)
        label = Text("Normal Distribution", font_size=24, color=WHITE)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg
        population_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        
        bell_group = VGroup(axes, bell_curve)
        
        # Apply positioning constraints
        self.place_in_area(bell_group, 'B2', 'E5', scale_factor=0.6)
        self.place_in_area(peak_line, 'C4', 'E4', scale_factor=0.7)
        self.place_in_area(label, 'A2', 'A3', scale_factor=0.7)
        self.place_at_grid(population_icon, 'C4', scale_factor=0.5)
        
        # Hide objects initially
        bell_group.set_opacity(0)
        peak_line.set_opacity(0)
        population_icon.set_opacity(0)
        label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#4287f5")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#4287f5")
        self.play(FadeIn(bell_group), FadeIn(label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#4287f5")
        self.play(bell_curve.animate.set_stroke(width=6), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(FadeIn(peak_line), FadeIn(population_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#4287f5")
        self.play(Indicate(bell_group, color="#4287f5", scale_factor=1.05))
        self.wait(2)
