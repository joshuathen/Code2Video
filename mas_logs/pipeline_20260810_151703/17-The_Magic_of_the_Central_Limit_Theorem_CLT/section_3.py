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
        self.setup_layout("The Reveal: Convergence to the Bell Curve", [
            "Chaotic data morphs into a Bell Curve.",
            "As sample sizes grow, patterns emerge naturally.",
            "This is the Central Limit Theorem revealed.",
            "Order emerges directly from random fluctuations.",
            "A perfect Normal Distribution finally forms."
        ])
        
        # Setup chaos/normal curve
        axes = Axes(x_range=[-4, 4], y_range=[0, 1], axis_config={"include_numbers": False}).scale(0.6)
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.7)
        
        # Simulate chaos (random points)
        chaos = VGroup(*[Dot(axes.c2p(np.random.uniform(-3, 3), np.random.uniform(0, 0.5)), radius=0.03, color=GRAY) for _ in range(100)])
        
        # The Bell Curve
        bell_curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#00CCFF", stroke_width=4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00CCFF")
        self.add(axes, chaos)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CCFF")
        self.play(FadeOut(chaos), Create(bell_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00CCFF")
        label = Text("Normal Distribution: Sampling Mean", font_size=20, color=WHITE)
        self.place_at_grid(label, 'B2', scale_factor=0.9)
        self.play(Write(label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CCFF")
        # Integrating [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        try:
            peak_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            peak_icon = Circle(radius=0.1, color=YELLOW, fill_opacity=1)
        
        peak_icon.move_to(axes.c2p(0, 0.4))
        self.play(Indicate(peak_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00CCFF")
        self.play(bell_curve.animate.set_stroke(width=6), run_time=1)
        self.wait(2)
