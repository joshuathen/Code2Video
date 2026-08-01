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
        # 1. Basic Requirements
        lines = [
            'Mathematics quantifies the surprise factor of each feedback pattern.', 
            'Rare patterns provide more information than common ones.', 
            'Entropy calculates the weighted average of these surprises.'
        ]
        self.setup_layout("The Math: Probability meets Information", lines)

        # === Animation for Lecture Line 1 ===
        # Highlight line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Formula construction: H = Σ p(i) log₂(1/p(i))
        # Note: Using Unicode for Σ (\u03a3) and log₂ (\u2082)
        f_h = Text("H = ", font_size=32, color=WHITE)
        f_sum = Text("\u03a3 p(i) ", font_size=32, color=WHITE)
        f_log = Text("log\u2082(1/p(i))", font_size=32, color=WHITE)
        formula = VGroup(f_h, f_sum, f_log).arrange(RIGHT, buff=0.1)
        
        # Mandatory Positioning: A1-B6 for formula
        # Resolved Issue #47: Scale factor 0.8
        self.place_in_area(formula, "A1", "B6", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            f_log.animate.set_color("#FFFF00") # Highlight log term as 'Surprise'
        )
        
        # Label for 'Surprise' component
        surprise_label = Text("Surprise", font_size=20, color="#FFFF00")
        surprise_label.next_to(f_log, DOWN, buff=0.3)
        
        # Visualizing Surprise vs Probability: Common vs Rare
        # Common: High p(i), Low Surprise
        bar_common = Rectangle(width=0.8, height=0.6, fill_opacity=0.8, color=WHITE, stroke_width=2)
        lbl_common = Text("Common\n(High p)", font_size=16, color=WHITE)
        group_common = VGroup(bar_common, lbl_common).arrange(DOWN, buff=0.2)
        # Resolved Issue #48: Moved to E2
        self.place_at_grid(group_common, "E2", scale_factor=1.0)
        
        # Rare: Low p(i), High Surprise
        bar_rare = Rectangle(width=0.8, height=2.4, fill_opacity=0.8, color="#FFFF00", stroke_width=2)
        lbl_rare = Text("Rare\n(Low p)", font_size=16, color="#FFFF00")
        group_rare = VGroup(bar_rare, lbl_rare).arrange(DOWN, buff=0.2)
        # Resolved Issue #49: Moved to E5
        self.place_at_grid(group_rare, "E5", scale_factor=1.0)

        self.play(
            FadeIn(surprise_label),
            FadeIn(group_common),
            FadeIn(group_rare)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Show Average Bits as a weighted balance line
        # Use C1-C6 row for the horizontal line
        avg_line = Line(start=self.grid["C1"] + LEFT*0.5, end=self.grid["C6"] + RIGHT*0.5, color="#00FF00", stroke_width=4)
        # Shift line slightly to represent the "average" height between the short and tall bar
        avg_line.shift(DOWN * 0.4) 
        
        avg_label = Text("Average Bits (Entropy)", font_size=22, color="#00FF00")
        avg_label.next_to(avg_line, UP, buff=0.1)
        
        self.play(
            Create(avg_line),
            Write(avg_label),
            Indicate(f_sum, color="#00FF00") # Highlight sum term to show weighted concept
        )
        self.wait(2)
