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

class Section5Scene(TeachingScene):
    def construct(self):
        # Colors
        GOLD = "#FFD700"
        SUM_COLOR = "#87CEEB"  # Sky Blue
        DIST_COLOR = "#98FB98" # Pale Green
        WHITE = "#FFFFFF"

        title = "The Grand Paradox: 1 + 2 + 4 + 8 + ... = -1"
        lines = [
            "The geometric series formula gives a surprising result: negative one.",
            "Partial sums like 1, 3, 7, and 15 appear.",
            "Each sum gets 2-adically closer to the number -1.",
            "The distance to -1 is always a power of two.",
            "In the 2-adic world, this infinite sum truly equals -1."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display 1 + 2 + 4 +... = -1 in gold #FFD700.
        self.lecture[0].set_color(GOLD)
        formula = MathTex(
            r"1 + 2 + 4 + 8 + \dots", r"=", r"\frac{1}{1-2}", r"=", r"-1",
            color=GOLD
        )
        self.place_in_area(formula, "A1", "A6", scale_factor=0.85)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show partial sums 1, 3, 7, 15, 31 appearing sequentially.
        # Resolves Issue 28: Potential overlap and poor spacing.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SUM_COLOR)
        
        sum_label2 = MathTex(r"S_n: 1, 3, 7, 15, 31, \dots", color=SUM_COLOR)
        self.place_in_area(sum_label2, 'B1', 'B6', scale_factor=0.8)
        self.play(FadeIn(sum_label2, shift=UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Each sum gets 2-adically closer to the number -1.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(DIST_COLOR)
        
        dist_label = MathTex(r"|S_n - (-1)|_2", r"=", r"|S_n + 1|_2", color=DIST_COLOR)
        self.place_in_area(dist_label, "C1", "C6", scale_factor=0.8)
        self.play(Write(dist_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The distance to -1 is always a power of two.
        # Animate the distance values getting smaller toward zero.
        # Resolves Issue 29: Missing grid assignment and visual gap.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(DIST_COLOR)
        
        dist_vals = MathTex(
            r"|S_n + 1|_2: \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \dots", 
            color=DIST_COLOR
        )
        self.place_in_area(dist_vals, 'D1', 'D6', scale_factor=0.75)
        self.play(FadeIn(dist_vals, shift=UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # In the 2-adic world, this infinite sum truly equals -1.
        # Highlight the final result -1 as the limit point.
        # Resolves Issue 27: Vertical disconnect and empty Row E.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD)
        
        result_box = SurroundingRectangle(formula[4], color=WHITE, buff=0.1)
        self.play(Create(result_box))
        
        limit_text = MathTex(r"\sum_{k=0}^{\infty} 2^k = -1 \text{ in } \mathbb{Q}_2", color=GOLD)
        self.place_in_area(limit_text, 'E1', 'F6', scale_factor=0.9)
        self.play(Write(limit_text))
        
        self.wait(2)
