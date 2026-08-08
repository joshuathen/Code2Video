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
        lecture_lines = ["The CLT reveals a hidden pattern.", "Sample means always approach a Bell Curve.", "This holds regardless of population shape.", "More samples make the curve clearer.", "Averages magically form a symmetrical distribution."]
        self.setup_layout("The CLT Core Mechanism: The Power of Averages", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # \"Display text: 'Averaging samples stabilizes erratic data.' in #FFFFFF.\"\n
        text_point = Text("Averaging samples stabilizes erratic data.", font_size=24, color=WHITE)
        self.place_at_grid(text_point, 'B4', scale_factor=0.6)
        self.play(Write(text_point))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)
        self.play(FadeOut(text_point))

        # === Animation for Lecture Line 2 ===
        # Show multiple small [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg] sample means animating towards the center (#FFFF00).
        dice_icons = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg", color=YELLOW) for _ in range(10)])
        for icon in dice_icons:
            start_pos = np.array([np.random.uniform(2, 6), np.random.uniform(-2, 2), 0])
            icon.move_to(start_pos)
        self.add(dice_icons)
        self.play(
            *[icon.animate.move_to(self.grid['C4']).scale(0.3) for icon in dice_icons],
            run_time=2
        )
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)
        self.play(FadeOut(dice_icons))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transition into a single larger Normal distribution shape (#00FF00) represented by [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scales.svg].
        scales_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scales.svg", color=GREEN)
        bell_curve = FunctionGraph(
            lambda x: np.exp(-(x**2)/2) / np.sqrt(2 * np.pi),
            x_range=[-3, 3],
            color=GREEN
        )
        # Using fix from issue 43: D3 to F6, scale 0.45
        self.place_in_area(bell_curve, 'D3', 'F6', scale_factor=0.45)
        self.place_at_grid(scales_icon, 'C3', scale_factor=0.8)
        
        self.play(Create(bell_curve), FadeIn(scales_icon))
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.wait(2)
