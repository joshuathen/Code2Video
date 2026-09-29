from manim import *
import numpy as np
from scipy.stats import binom

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Visualization", [
            "Distribution shape depends on n and p.", 
            "Changing p morphs the histogram shape.", 
            "Binomial distributions predict real-world likelihoods."
        ])
        
        # Assets
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        # Data for binomial(n=5, p=0.7)
        n = 5
        p = 0.7
        probs = [binom.pmf(k, n, p) for k in range(n + 1)]
        
        # Creating bars
        bars = VGroup(*[
            Rectangle(height=prob * 4, width=0.6, fill_opacity=0.7, color=BLUE)
            for prob in probs
        ]).arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
        
        # Label for Sum
        sum_label = Text("Total Probability = 1", font_size=20, color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(dice_icon, "A5", "A6", scale_factor=0.3)
        self.place_in_area(bars, "B3", "E6", scale_factor=0.6)
        self.play(FadeIn(dice_icon), Create(bars))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        
        # Morphing p from 0.7 to 0.1, then 0.9
        for new_p in [0.1, 0.9]:
            new_probs = [binom.pmf(k, n, new_p) for k in range(n + 1)]
            new_bars = VGroup(*[
                Rectangle(height=prob * 4, width=0.6, fill_opacity=0.7, color=BLUE)
                for prob in new_probs
            ]).arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
            self.place_in_area(new_bars, "B3", "E6", scale_factor=0.6)
            
            self.play(Transform(bars, new_bars))
            
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.place_in_area(coin_icon, "A5", "A6", scale_factor=0.3)
        self.place_at_grid(sum_label, "F6", scale_factor=0.7)
        
        # Highlight total area under all bars in color #32CD32
        for bar in bars:
            bar.set_color("#32CD32")
        
        self.play(FadeOut(dice_icon), FadeIn(coin_icon), Write(sum_label))
        self.wait(2)
