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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Core Mechanism: Entropy in Information Theory", [
            "Wordle is about maximizing information gain.",
            "Each guess is a tool to gather info.",
            "Think of the word pool as possibilities.",
            "A good guess splits the search tree.",
            "We want to narrow candidates effectively."
        ])
        
        # --- Animation components ---
        # Bar chart
        bars = VGroup(*[Rectangle(height=np.random.rand() * 2 + 0.5, width=0.5, color=GOLD) for _ in range(5)])
        bars.arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        self.place_in_area(bars, "B3", "D6", scale_factor=0.7)
        
        # Bits tracker
        bits_label = Text("Entropy: 5.0 bits", font_size=24, color=WHITE)
        self.place_at_grid(bits_label, "F3", scale_factor=0.8)
        
        # Asset
        wordle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wordle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD)
        self.play(FadeIn(bars))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GOLD)
        self.play(Write(bits_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GOLD)
        self.play(*[bar.animate.set_height(np.random.rand() * 1.5 + 0.5) for bar in bars])

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GOLD)
        new_bits_text = Text("Entropy: 2.1 bits", font_size=24, color=WHITE)
        new_bits_text.move_to(bits_label.get_center())
        self.play(
            FadeTransform(bits_label, new_bits_text),
            *[bar.animate.set_height(np.random.rand() * 0.8 + 0.2) for bar in bars]
        )
        bits_label = new_bits_text

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD)
        final_bits = Text("Entropy: 0.5 bits", font_size=24, color=GREEN)
        self.place_at_grid(final_bits, "F4", scale_factor=0.9)
        self.place_at_grid(wordle_icon, "F2", scale_factor=0.5)
        
        self.play(
            FadeTransform(bits_label, final_bits),
            FadeIn(wordle_icon),
            *[bar.animate.set_height(0.2) for bar in bars]
        )
        self.wait(2)
