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
        # Fetch data from storyboard/outline
        title_text = "The Discrete Case: The 'Sum of Dice' Logic"
        lecture_lines = [
            "Consider rolling two dice to get a total of four.",
            "We could roll a one and a three.",
            "Or two and two, or three and one.",
            "We sum all combinations that yield our target total.",
            "This discrete sum is the heart of convolution logic."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        C_L1 = WHITE
        C_L2 = "#ADD8E6" # Light Blue
        C_L3 = "#90EE90" # Light Green
        C_L4 = "#FFFF00" # Yellow
        C_L5 = "#FFA500" # Orange

        # Asset path from Issue 24
        dice_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg"

        # Helper for dice (used for specific values)
        def create_dice(val, color=WHITE):
            dice = VGroup()
            square = Square(side_length=0.4, color=color, stroke_width=2)
            dots = VGroup()
            dot_positions = {
                1: [[0, 0, 0]],
                2: [[-0.1, 0.1, 0], [0.1, -0.1, 0]],
                3: [[-0.1, 0.1, 0], [0, 0, 0], [0.1, -0.1, 0]],
                4: [[-0.1, 0.1, 0], [0.1, 0.1, 0], [-0.1, -0.1, 0], [0.1, -0.1, 0]],
                5: [[-0.1, 0.1, 0], [0.1, 0.1, 0], [0, 0, 0], [-0.1, -0.1, 0], [0.1, -0.1, 0]],
                6: [[-0.1, 0.1, 0], [0.1, 0.1, 0], [-0.1, 0, 0], [0.1, 0, 0], [-0.1, -0.1, 0], [0.1, -0.1, 0]],
            }
            for pos in dot_positions.get(val, []):
                dots.add(Dot(radius=0.03, color=color).move_to(square.get_center() + np.array(pos)))
            dice.add(square, dots)
            return dice

        # === Animation for Lecture Line 1 ===
        # "Consider rolling two dice to get a total of four."
        self.lecture[0].set_color(C_L1)
        # Load asset-based dice for the general case
        dice1 = SVGMobject(dice_asset_path, color=WHITE)
        dice2 = SVGMobject(dice_asset_path, color=WHITE)
        self.place_at_grid(dice1, "A1", scale_factor=0.6)
        self.place_at_grid(dice2, "A2", scale_factor=0.6)
        
        target_sum_label = Text("Target Sum: 4", font_size=24, color=C_L1)
        # Fix: Layout fix 1 from VideoCritic (Issue 41)
        self.place_in_area(target_sum_label, 'A3', 'A6', scale_factor=0.8)
        
        self.play(FadeIn(dice1), FadeIn(dice2), Write(target_sum_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We could roll a one and a three."
        self.lecture[1].set_color(C_L2)
        comb1 = VGroup(create_dice(1, C_L2), Text("+", font_size=20), create_dice(3, C_L2), Text("= 4", font_size=20)).arrange(RIGHT, buff=0.1)
        # Fix: Layout fix 3 from VideoCritic (Issue 41)
        self.place_at_grid(comb1, 'B1', scale_factor=0.7)
        self.play(FadeIn(comb1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Or two and two, or three and one."
        self.lecture[2].set_color(C_L3)
        comb2 = VGroup(create_dice(2, C_L3), Text("+", font_size=20), create_dice(2, C_L3), Text("= 4", font_size=20)).arrange(RIGHT, buff=0.1)
        comb3 = VGroup(create_dice(3, C_L3), Text("+", font_size=20), create_dice(1, C_L3), Text("= 4", font_size=20)).arrange(RIGHT, buff=0.1)
        # Fix: Layout fix 3 from VideoCritic (Issue 41)
        self.place_at_grid(comb2, 'C1', scale_factor=0.7)
        self.place_at_grid(comb3, 'D1', scale_factor=0.7)
        self.play(FadeIn(comb2), FadeIn(comb3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We sum all combinations that yield our target total."
        self.lecture[3].set_color(C_L4)
        
        # Clear items to make room for the table
        self.play(FadeOut(comb1), FadeOut(comb2), FadeOut(comb3), FadeOut(target_sum_label), FadeOut(dice1), FadeOut(dice2))
        
        # Display a 6x6 table of possible sums (2 to 12)
        table_vals = [[r + c for c in range(1, 7)] for r in range(1, 7)]
        table_group = VGroup()
        for i, row in enumerate(table_vals):
            for j, val in enumerate(row):
                cell = Text(str(val), font_size=16, color=WHITE)
                table_group.add(cell)
        
        table_group.arrange_in_grid(rows=6, cols=6, buff=0.5)
        self.place_in_area(table_group, "B1", "F6", scale_factor=1.0)
        
        # Highlight sums of 4
        highlight_4s = VGroup()
        for i, row in enumerate(table_vals):
            for j, val in enumerate(row):
                if val == 4:
                    rect = SurroundingRectangle(table_group[i*6 + j], color=C_L4, buff=0.05)
                    highlight_4s.add(rect)

        self.play(FadeIn(table_group))
        self.play(Create(highlight_4s))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This discrete sum is the heart of convolution logic."
        self.lecture[4].set_color(C_L5)
        
        # Identify indices for sum '7'
        target_7_indices = [i*6 + j for i, row in enumerate(table_vals) for j, val in enumerate(row) if val == 7]
        target_7_cells = VGroup(*[table_group[idx] for idx in target_7_indices])
        
        highlight_7s = VGroup()
        for cell in target_7_cells:
            rect = SurroundingRectangle(cell, color=C_L4, buff=0.05)
            highlight_7s.add(rect)
        
        self.play(FadeOut(highlight_4s))
        # Visual cue for the "most frequent outcome" logic from storyboard
        self.play(Flash(target_7_cells, color=C_L4, flash_radius=0.3))
        self.play(Create(highlight_7s))
        
        # Convolution formula
        formula = MathTex(r"P(X+Y=s) = \sum_{k} P(X=k)P(Y=s-k)", font_size=28, color=C_L5)
        # Fix: Layout fix 2 from VideoCritic (Issue 41)
        self.place_in_area(formula, 'A3', 'A6', scale_factor=0.6)
        
        self.play(Write(formula))
        self.wait(2)
