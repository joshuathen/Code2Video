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
        self.setup_layout("Critical Knowledge Points", [
            "Sample means always approach normality.",
            "Original distribution shape doesn't matter.",
            "Sample size thirty ensures significance."
        ])
        
        # Assets
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coins.svg")

        # === Animation for Lecture Line 1 ===
        n5_text = Text("n=5", color="#FF3333", font_size=24)
        self.place_at_grid(n5_text, "A2", scale_factor=0.9)
        
        n30_text = Text("n=30", color="#33FF57", font_size=24)
        self.place_at_grid(n30_text, "A5", scale_factor=0.9)
        
        dice_n5 = dice_icon.copy()
        self.place_at_grid(dice_n5, "B2", scale_factor=0.5)
        
        dice_n30 = dice_icon.copy()
        self.place_at_grid(dice_n30, "B5", scale_factor=0.5)
        
        curve_n5 = FunctionGraph(lambda x: 0.5 * np.sin(5*x) * np.exp(-x**2), x_range=[-2, 2], color="#FF3333")
        self.place_in_area(curve_n5, "B1", "C3", scale_factor=0.6)
        
        curve_n30 = FunctionGraph(lambda x: 1.0 * np.exp(-x**2), x_range=[-2, 2], color="#33FF57")
        self.place_in_area(curve_n30, "B4", "C6", scale_factor=0.6)
        
        self.play(FadeIn(n5_text), FadeIn(dice_n5), Create(curve_n5), FadeIn(n30_text), FadeIn(dice_n30), Create(curve_n30))
        self.lecture[0].set_color("#FF3333")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        stability_bar = Rectangle(height=0.5, width=4.0, color=WHITE)
        self.place_at_grid(stability_bar, "E2", scale_factor=1.0)
        
        self.play(Create(stability_bar))
        self.lecture[1].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        n_ge_30 = Text("n >= 30", color="#FFFFFF", font_size=32)
        self.place_at_grid(n_ge_30, "E5", scale_factor=1.0)
        
        coin = coin_icon.copy()
        self.place_at_grid(coin, "F5", scale_factor=0.5)
        
        self.play(Flash(n_ge_30), Write(n_ge_30), GrowFromCenter(coin))
        self.lecture[2].set_color("#FFFF33")
        self.wait(2)
