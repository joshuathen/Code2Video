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
        self.setup_layout("Calculating Entropy: Expected Information Gain", [
            "Entropy calculates the average information in guesses.",
            "Guessing splits possibilities into equal-sized buckets.",
            "High entropy choices resolve uncertainty fastest.",
            "We average information gain across all outcomes.",
            "Formula sums probability times surprise for all."
        ])
        
        # 1. Display Probability Distribution Histogram
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg]
        hist = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        hist.set_color("#FF00FF")
        self.place_in_area(hist, 'C4', 'F6', scale_factor=0.6)
        self.play(FadeIn(hist))
        self.lecture[0].set_color("#FF00FF")
        self.wait(4)

        # 2. Overlay Entropy formula
        # Labeling P(x) and I(x)
        formula = MathTex(r"H(X) = \sum P(x)I(x)", color="#FFFFFF")
        px_label = MathTex(r"P(x)", color="#FF00FF").next_to(formula, DOWN)
        ix_label = MathTex(r"I(x)", color="#FFFF00").next_to(px_label, RIGHT)
        labels = VGroup(px_label, ix_label)
        
        formula_group = VGroup(formula, labels)
        self.place_at_grid(formula_group, 'B4', scale_factor=0.7)
        self.play(Write(formula), FadeIn(labels))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(4)

        # 3. Highlight parts
        self.play(Indicate(px_label), Indicate(ix_label))
        self.lecture[2].set_color(YELLOW)
        self.wait(4)

        # 4. Place Goal Anchor
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg]
        target = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        target.set_color(GREEN)
        self.place_at_grid(target, 'E5', scale_factor=0.5)
        self.play(FadeIn(target))
        self.lecture[3].set_color(GREEN)
        self.wait(4)

        # 5. Pulse the anchor
        self.play(Indicate(target))
        self.lecture[4].set_color(GREEN)
        self.wait(4)
