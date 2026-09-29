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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Uncertainty Principle: Duality", [
            "Define spread in time and frequency.",
            "Observe the inequality: delta t times delta omega.",
            "Squeeze time, and frequency spreads out.",
            "Gaussian pulses visualize this trade-off.",
            "Mathematical bounds are unavoidable."
        ])
        
        # Assets
        sinusoid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sinusoid.svg")
        
        # Elements
        t_axis = Line(LEFT*2, RIGHT*2, color=BLUE)
        w_axis = Line(DOWN*1.5, UP*1.5, color=GREEN)
        
        time_gauss = FunctionGraph(lambda t: np.exp(-t**2 * 5), x_range=[-2, 2], color=YELLOW)
        freq_gauss = FunctionGraph(lambda w: np.exp(-w**2 * 0.2), x_range=[-4, 4], color=RED)
        
        pulse_group = VGroup(t_axis, time_gauss, w_axis, freq_gauss, sinusoid_icon)
        
        # Apply the fix for issue 24/39: place in area 'A1'-'C3' with 0.6 scale
        self.place_in_area(pulse_group, 'A1', 'C3', scale_factor=0.6)

        # Apply the fix for issue 26/41: Grid labels at 'D4' with 0.7 scale
        grid_labels = Text("T-F Trade-off", font_size=20)
        self.place_at_grid(grid_labels, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(t_axis), Create(w_axis), Create(sinusoid_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(Create(time_gauss), Create(freq_gauss))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(Write(grid_labels))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        self.wait(1)
