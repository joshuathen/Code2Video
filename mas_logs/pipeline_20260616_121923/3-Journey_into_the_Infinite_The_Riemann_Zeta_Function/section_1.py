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
        # Initialize layout with the required title and script lines
        lecture_lines = [
            "Infinite sums don't always reach infinity.",
            "Zeno’s rabbit jumps halfway, then half again.",
            "He approaches the wall but never crosses."
        ]
        self.setup_layout("The Infinite Staircase: Prerequisite Knowledge", lecture_lines)

        # Define anchor points based on grid for consistent layout
        start_pt = self.grid['D1']
        end_pt = self.grid['D6']

        # === Animation for Lecture Line 1 ===
        # Highlight first line in white
        self.lecture[0].set_color(WHITE)
        
        # White horizontal line labeled '0' and '1'
        num_line = Line(start_pt, end_pt, color=WHITE, stroke_width=4)
        label_0 = Text("0", font_size=24, color=WHITE)
        label_1 = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(label_0, "E1") # Positioned below start
        self.place_at_grid(label_1, "E6") # Positioned below end
        
        # Wall Asset
        wall = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wall.svg")
        self.place_at_grid(wall, "D6", scale_factor=0.4)
        wall.shift(RIGHT * 0.3) # Offset slightly to mark the boundary clearly
        
        self.play(Create(num_line), Write(label_0), Write(label_1), FadeIn(wall))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in green
        self.lecture[1].set_color(GREEN)
        
        # Zeno the Rabbit Asset
        zeno = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/rabbit.svg")
        zeno.set_color(GREEN)
        self.place_at_grid(zeno, "D1", scale_factor=0.3)
        
        # Key coordinates for convergence
        mid_pt = (start_pt + end_pt) / 2
        
        # Jump 1: To 1/2
        jump1 = ArcBetweenPoints(start_pt, mid_pt, radius=-1.5, color=GREEN)
        label_half = Text("1/2", font_size=20, color=GREEN)
        # Resolved Issue 35: Updated position to E3 and scale to 0.9
        self.place_at_grid(label_half, 'E3', scale_factor=0.9)
        
        self.play(FadeIn(zeno))
        self.play(MoveAlongPath(zeno, jump1), run_time=1.2)
        self.play(Write(label_half))
        
        # === Animation for Lecture Line 3 ===
        # Highlight third line in red
        self.lecture[2].set_color(RED)
        
        # Jump 2: To 3/4 (adding 1/4)
        three_fourth_pt = (mid_pt + end_pt) / 2
        jump2 = ArcBetweenPoints(mid_pt, three_fourth_pt, radius=-0.8, color=GREEN)
        label_quarter = Text("1/4", font_size=18, color=GREEN)
        # Resolved Issue 36: Updated position to E4 and scale to 0.8
        self.place_at_grid(label_quarter, 'E4', scale_factor=0.8)
        
        self.play(MoveAlongPath(zeno, jump2), run_time=1)
        self.play(Write(label_quarter))
        
        # Jump 3: To 7/8 (adding 1/8)
        seven_eighth_pt = (three_fourth_pt + end_pt) / 2
        jump3 = ArcBetweenPoints(three_fourth_pt, seven_eighth_pt, radius=-0.4, color=GREEN)
        label_eighth = Text("1/8", font_size=16, color=GREEN)
        # Resolved Issue 34: Updated position to E5 and scale to 0.8
        self.place_at_grid(label_eighth, 'E5', scale_factor=0.8)
        
        self.play(MoveAlongPath(zeno, jump3), run_time=0.8)
        self.play(Write(label_eighth))
        
        # Red arrow highlighting the gap to 1
        remaining_gap_arrow = Arrow(
            start=seven_eighth_pt + UP*0.3, 
            end=end_pt + UP*0.3, 
            color=RED, 
            buff=0.05, 
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3
        )
        
        self.play(Create(remaining_gap_arrow))
        self.wait(2)
