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
        # Initialize title and lecture lines
        title_str = "The Pattern Approach: Dividing Down"
        lines = [
            "Let's look at factorials from a different perspective.",
            "To find the previous factorial, just divide by n.",
            "Our factorial elevator descends by dividing at each floor.",
            "One factorial divided by one equals one.",
            "Mathematically, this shows why zero factorial must be one."
        ]
        self.setup_layout(title_str, lines)

        # Colors
        GOLD_COLOR = "#FFD700"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Let's look at factorials from a different perspective.
        self.lecture[0].set_color(YELLOW)
        
        # Elevator shaft (visual side bar)
        shaft = Line(self.grid["A3"], self.grid["F3"], color=GRAY_A)
        values = ["4! = 24", "3! = 6", "2! = 2", "1! = 1", "0! = ?"]
        level_labels = VGroup()
        
        grid_keys = ["A4", "B4", "C4", "D4", "E4"]
        for i, (val, key) in enumerate(zip(values, grid_keys)):
            # Changed MathTex to Text to avoid FileNotFoundError: 'latex'
            label = Text(val, font_size=28)
            self.place_at_grid(label, key)
            level_labels.add(label)
        
        # Elevator Car
        elevator_car = RoundedRectangle(height=0.6, width=2.0, corner_radius=0.1, color=WHITE_COLOR)
        elevator_car.move_to(level_labels[0].get_center())
        
        self.play(Create(shaft), Write(level_labels), Create(elevator_car))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # To find the previous factorial, just divide by n.
        self.lecture[1].set_color(YELLOW)
        
        # Show movement from 4! to 3!
        arrow1 = Arrow(level_labels[0].get_bottom(), level_labels[1].get_top(), buff=0.1, color=WHITE_COLOR)
        # Using Unicode division symbol to replace \div
        div_label1 = Text("÷ 4", font_size=24, color=WHITE_COLOR).next_to(arrow1, RIGHT)
        
        self.play(
            elevator_car.animate.move_to(level_labels[1].get_center()),
            GrowArrow(arrow1),
            Write(div_label1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Our factorial elevator descends by dividing at each floor.
        self.lecture[2].set_color(YELLOW)
        
        # Move from 3! to 2!
        arrow2 = Arrow(level_labels[1].get_bottom(), level_labels[2].get_top(), buff=0.1, color=WHITE_COLOR)
        div_label2 = Text("÷ 3", font_size=24, color=WHITE_COLOR).next_to(arrow2, RIGHT)
        
        self.play(
            elevator_car.animate.move_to(level_labels[2].get_center()),
            GrowArrow(arrow2),
            Write(div_label2)
        )
        self.wait(0.5)
        
        # Move from 2! to 1!
        arrow3 = Arrow(level_labels[2].get_bottom(), level_labels[3].get_top(), buff=0.1, color=WHITE_COLOR)
        div_label3 = Text("÷ 2", font_size=24, color=WHITE_COLOR).next_to(arrow3, RIGHT)
        
        self.play(
            elevator_car.animate.move_to(level_labels[3].get_center()),
            GrowArrow(arrow3),
            Write(div_label3)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # One factorial divided by one equals one.
        self.lecture[3].set_color(YELLOW)
        
        # Move from 1! to 0!
        arrow4 = Arrow(level_labels[3].get_bottom(), level_labels[4].get_top(), buff=0.1, color=WHITE_COLOR)
        div_label4 = Text("÷ 1", font_size=24, color=WHITE_COLOR).next_to(arrow4, RIGHT)
        
        # Prepare the final value
        final_zero_fact = Text("0! = 1", font_size=28, color=GOLD_COLOR)
        self.place_at_grid(final_zero_fact, "E4")
        
        self.play(
            elevator_car.animate.move_to(level_labels[4].get_center()),
            GrowArrow(arrow4),
            Write(div_label4)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Mathematically, this shows why zero factorial must be one.
        self.lecture[4].set_color(YELLOW)
        
        self.play(
            Transform(level_labels[4], final_zero_fact),
            elevator_car.animate.set_color(GOLD_COLOR)
        )
        self.play(Indicate(level_labels[4]))
        self.wait(2)
