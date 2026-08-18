from manim import *
import os

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
        self.setup_layout("The Problem: Context and Ambiguity", [
            "Words change meaning based on surrounding context.",
            "Self-attention weighs importance of every other word.",
            "It connects 'it' to the right antecedent."
        ])
        
        # Hide lecture lines initially
        for line in self.lecture:
            line.set_opacity(0)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bank.svg
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bank.svg"
        bank_icon = SVGMobject(asset_path).set_color(YELLOW)
        self.place_at_grid(bank_icon, 'C3', scale_factor=0.8)
        
        sentence = Text("I went to the bank to deposit money.", font_size=24, color=WHITE)
        self.place_at_grid(sentence, 'B2', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), FadeIn(sentence))
        self.play(self.lecture[0].animate.set_color(YELLOW), FadeIn(bank_icon))
        
        # Highlight "bank"
        bank_word = sentence[16:20]
        bank_box = SurroundingRectangle(bank_word, color=RED, buff=0.1)
        self.play(Create(bank_box))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(self.lecture[2].animate.set_color(TEAL))
        self.wait(2)
