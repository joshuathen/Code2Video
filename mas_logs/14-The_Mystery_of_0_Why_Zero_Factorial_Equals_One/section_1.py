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
        # Initial Setup
        title_text = "Introduction: The Factorial Machine"
        lines = [
            "Meet Facto, our machine that calculates factorials.",
            "Factorials multiply all integers down to one.",
            "Three factorial is three times two times one.",
            "But what happens when Facto reaches zero?",
            "Zero factorial seems like a mystery."
        ]
        self.setup_layout(title_text, lines)

        # Asset path
        robot_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        # Meet Facto, our machine that calculates factorials.
        self.lecture[0].set_color("#FFD700")
        
        # Subtitle
        subtitle = Text("The Mystery of 0!", color=WHITE, font_size=24)
        self.place_in_area(subtitle, "A1", "A6")
        
        # Facto Robot Design with Asset
        facto_body = SVGMobject(robot_path).set_color("#FFD700")
        facto_label = Text("Facto", color="#FFD700", font_size=20)
        facto = VGroup(facto_body, facto_label).arrange(DOWN, buff=0.1)
        self.place_in_area(facto, "C3", "D4", scale_factor=1.2)
        
        self.play(Write(subtitle))
        self.play(FadeIn(facto))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Factorials multiply all integers down to one.
        self.lecture[1].set_color(WHITE)
        formula = Text("n! = n × (n-1) × ... × 1", color=WHITE)
        self.place_in_area(formula, "B1", "B6", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Three factorial is three times two times one.
        self.lecture[2].set_color("#00FF00")
        
        # 3! animation - Issue 29: Input at D2
        three_fact = Text("3!", color="#00FF00", font_size=36)
        self.place_at_grid(three_fact, "D2", scale_factor=0.8)
        
        self.play(FadeIn(three_fact))
        self.play(three_fact.animate.move_to(facto.get_center()))
        
        # Issue 28: internal process at E3-E4
        process_3 = Text("3 × 2 × 1", color="#00FF00", font_size=28)
        self.place_in_area(process_3, "E3", "E4", scale_factor=0.8) 
        
        self.play(ReplacementTransform(three_fact, process_3))
        self.wait(0.5)
        
        # Issue 30: Output at D5
        out_6 = Text("6", color="#00FF00", font_size=36)
        self.place_at_grid(out_6, "D5", scale_factor=0.8)
        
        self.play(ReplacementTransform(process_3, out_6))
        self.play(out_6.animate.shift(RIGHT * 0.5), run_time=0.5)
        self.play(FadeOut(out_6))
        
        # === Animation for Lecture Line 4 ===
        # But what happens when Facto reaches zero?
        
        # 2! - Issue 29 and 30
        two_fact = Text("2!", color="#00FF00", font_size=32)
        self.place_at_grid(two_fact, "D2", scale_factor=0.8)
        out_2 = Text("2", color="#00FF00", font_size=32)
        self.place_at_grid(out_2, "D5", scale_factor=0.8)
        
        self.play(FadeIn(two_fact))
        self.play(two_fact.animate.move_to(facto.get_center()), run_time=0.4)
        self.play(ReplacementTransform(two_fact, out_2), run_time=0.4)
        self.play(FadeOut(out_2), run_time=0.4)
        
        # 1! - Issue 29 and 30
        one_fact = Text("1!", color="#00FF00", font_size=32)
        self.place_at_grid(one_fact, "D2", scale_factor=0.8)
        out_1 = Text("1", color="#00FF00", font_size=32)
        self.place_at_grid(out_1, "D5", scale_factor=0.8)
        
        self.play(FadeIn(one_fact))
        self.play(one_fact.animate.move_to(facto.get_center()), run_time=0.4)
        self.play(ReplacementTransform(one_fact, out_1), run_time=0.4)
        self.play(FadeOut(out_1), run_time=0.4)
        
        # Transition to 0! - Issue 29
        self.lecture[3].set_color("#FF0000")
        zero_fact = Text("0!", color="#FF0000", font_size=40)
        self.place_at_grid(zero_fact, "D2", scale_factor=0.8)
        
        self.play(FadeIn(zero_fact))
        self.play(zero_fact.animate.move_to(facto.get_center()))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Zero factorial seems like a mystery.
        self.lecture[4].set_color("#FF0000")
        
        # Confusion state: Replace eyes or overlay question marks
        q_marks = Text("??", color="#FF0000", font_size=48)
        q_marks.move_to(facto_body.get_center())
        
        self.play(
            FadeIn(q_marks),
            facto_body.animate.set_color("#FF0000"),
            facto_label.animate.set_color("#FF0000")
        )
        self.wait(2)
