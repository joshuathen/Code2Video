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
        self.setup_layout("Concrete Application: Sound Synthesis", [
            "Sound waves are vector sums.",
            "Bird songs combine as frequency vectors.",
            "Audio engineering uses this linear combination."
        ])
        
        # Load assets
        bird = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        
        # Animation 1: Sine wave, color #00FFFF, bird flying along
        wave = FunctionGraph(lambda x: 0.5 * np.sin(x * 3), x_range=[-3, 3], color="#00FFFF")
        self.place_in_area(wave, "A4", "C6", scale_factor=0.6)
        
        bird1 = bird.copy().scale(0.3)
        bird1.move_to(wave.get_start())
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(wave), FadeIn(bird1), self.lecture[0].animate.set_color("#00FFFF"))
        self.play(MoveAlongPath(bird1, wave), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        formula = MathTex(r"v_{bird} = f_A + f_B", color="#FFFF00")
        self.place_at_grid(formula, "D4", scale_factor=0.9)
        self.play(Write(formula), self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        new_wave = FunctionGraph(lambda x: 0.8 * np.sin(x * 2), x_range=[-3, 3], color="#FF0000")
        self.place_in_area(new_wave, "D1", "F6", scale_factor=0.7)
        
        bird2 = bird.copy().scale(0.3)
        bird2.move_to(new_wave.get_start())
        
        self.play(ReplacementTransform(wave, new_wave), FadeOut(bird1), FadeIn(bird2), self.lecture[2].animate.set_color("#FF0000"))
        self.play(bird2.animate.scale(1.5), run_time=1)
        self.wait(2)
