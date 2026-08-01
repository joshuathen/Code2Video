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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines from storyboard
        title_text = "Prerequisite Check: PDF and Independence"
        lecture_lines = [
            "A PDF shows the probability of each outcome.",
            "Battery A's life doesn't affect battery B.",
            "These independent events have separate probability curves."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        GOLD = "#FFD700"
        GREEN = "#00FF00"
        YELLOW = "#FFFF00"
        CYAN = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Script: "A PDF shows the probability of each outcome."
        # Animation: Flash 'PDF: Shape of Uncertainty' text in gold (#FFD700).
        self.play(self.lecture[0].animate.set_color(GOLD))
        pdf_text = Text("PDF: Shape of Uncertainty", color=GOLD)
        # Issue 29 Fix: Use place_in_area for multi-word string and move to B2-B6
        self.place_in_area(pdf_text, 'B2', 'B6', scale_factor=0.8)
        self.play(FadeIn(pdf_text))
        self.play(Indicate(pdf_text, color=GOLD))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Script: "Battery A's life doesn't affect battery B."
        # Animation: Show independent bell curves for X and Y (#00FF00, #FFFF00).
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Define battery A bell curve group
        axes_a = Axes(
            x_range=[-2.5, 2.5], 
            y_range=[0, 1.2], 
            x_length=2.0, 
            y_length=1.5,
            axis_config={"include_tip": False}
        ).set_color(GREEN)
        curve_a = axes_a.plot(lambda x: np.exp(-x**2), color=GREEN)
        label_a = Text("Battery A", color=GREEN).scale(0.5).next_to(axes_a, DOWN, buff=0.1)
        group_a = VGroup(axes_a, curve_a, label_a)
        
        # Define battery B bell curve group
        axes_b = Axes(
            x_range=[-2.5, 2.5], 
            y_range=[0, 1.2], 
            x_length=2.0, 
            y_length=1.5,
            axis_config={"include_tip": False}
        ).set_color(YELLOW)
        curve_b = axes_b.plot(lambda x: np.exp(-x**2), color=YELLOW)
        label_b = Text("Battery B", color=YELLOW).scale(0.5).next_to(axes_b, DOWN, buff=0.1)
        group_b = VGroup(axes_b, curve_b, label_b)

        # Issue 30 Fix: Move to Row E for better grid utilization (away from row A/B labels)
        self.place_at_grid(group_a, "E3", scale_factor=0.8)
        self.place_at_grid(group_b, "E5", scale_factor=0.8)

        self.play(
            Create(axes_a),
            Create(curve_a),
            Write(label_a)
        )
        self.play(
            Create(axes_b),
            Create(curve_b),
            Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Script: "These independent events have separate probability curves."
        # Animation: Draw dashed line between curves labeled 'Independent' (#00FFFF).
        self.play(self.lecture[2].animate.set_color(CYAN))
        
        # Issue 31 Fix: Scale to 0.8 and move to D4 (between curves at Row E)
        independent_label = Text("Independent", color=CYAN)
        self.place_at_grid(independent_label, 'D4', scale_factor=0.8)

        # Create a connection with dashed line
        dashed_line = DashedLine(
            start=group_a.get_critical_point(RIGHT) + RIGHT*0.2,
            end=group_b.get_critical_point(LEFT) + LEFT*0.2,
            color=CYAN
        )
        
        self.play(Create(dashed_line))
        self.play(Write(independent_label))
        self.wait(2)
