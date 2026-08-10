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
        self.setup_layout("Defining Superposition via Dirac Notation", [
            "Superposition follows the state vector formula.",
            "State vector equals alpha times 0 plus beta times 1.",
            "Alpha and beta are complex probability amplitudes.",
            "A spinning coin embodies this combined state.",
            "The weights of 0 and 1 change dynamically."
        ])

        # --- Assets ---
        # Formula
        formula = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color=YELLOW)
        
        # Gauge Bars
        gauge0 = Rectangle(width=0.5, height=2, fill_opacity=0.8, color=BLUE)
        gauge1 = Rectangle(width=0.5, height=2, fill_opacity=0.8, color=RED)
        label0 = MathTex(r"|0\rangle")
        label1 = MathTex(r"|1\rangle")
        
        # Spinning coin (Asset integration)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # === Animation for Lecture Line 1 ===
        # Fixing issue 24: formula positioning
        self.place_in_area(formula, 'B3', 'B5', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Fixing issue 25: gauge/label positioning
        self.place_at_grid(gauge0, 'D3', scale_factor=0.5)
        self.place_at_grid(gauge1, 'D4', scale_factor=0.5)
        # Place labels closer to their respective gauges
        label0.next_to(gauge0, DOWN, buff=0.1)
        label1.next_to(gauge1, DOWN, buff=0.1)
        self.play(Create(gauge0), Create(gauge1), Write(label0), Write(label1))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Fixing issue 26: coin positioning
        self.place_at_grid(coin, 'E5', scale_factor=0.7)
        self.play(FadeIn(coin))
        self.play(Rotate(coin, angle=2*PI, run_time=2))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        # Animate weights
        tracker = ValueTracker(0.5)
        gauge0.add_updater(lambda m: m.set_height(2 * tracker.get_value()))
        gauge1.add_updater(lambda m: m.set_height(2 * (1 - tracker.get_value())))
        
        self.play(tracker.animate.set_value(0.8), run_time=2)
        self.play(tracker.animate.set_value(0.2), run_time=2)
        self.wait(1)
