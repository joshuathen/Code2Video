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
        title = "The Prerequisite: The Bernoulli Trial"
        lines = [
            "Every complex probability starts with a single simple trial.",
            "Meet Byte. He either makes the shot or misses.",
            "We call this a Bernoulli Trial: Success or Failure."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Bernoulli Toggle Switch
        switch_label = Text("Bernoulli Trial", font_size=24, color=WHITE)
        # Issue 22 Fix: Centering label over area A3-A4
        self.place_in_area(switch_label, "A3", "A4", scale_factor=0.8)

        switch_rect = RoundedRectangle(height=0.8, width=1.6, corner_radius=0.4, color=WHITE)
        # Issue 24 Fix: Scaling to 0.8 for consistency
        self.place_in_area(switch_rect, "B3", "B4", scale_factor=0.8)
        
        knob = Circle(radius=0.25, color=WHITE, fill_opacity=1)
        # Position knob inside the switch rect
        knob.move_to(switch_rect.get_left() + RIGHT * 0.35)
        
        success_label = Text("Success", color="#00FF00", font_size=20)
        failure_label = Text("Failure", color="#FF0000", font_size=20)
        self.place_at_grid(failure_label, "B2", scale_factor=0.8)
        self.place_at_grid(success_label, "B5", scale_factor=0.8)

        self.play(FadeIn(switch_label), FadeIn(switch_rect), FadeIn(knob), FadeIn(success_label), FadeIn(failure_label))
        
        # Flipping animation
        self.play(
            knob.animate.move_to(switch_rect.get_right() - RIGHT * 0.35),
            switch_rect.animate.set_fill("#00FF00", opacity=0.3),
            run_time=1
        )
        self.play(
            knob.animate.move_to(switch_rect.get_left() + RIGHT * 0.35),
            switch_rect.animate.set_fill("#FF0000", opacity=0.3),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Issue 20 Fix: Use Byte Asset
        byte = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        byte.set_color("#58C4DD")
        # Issue 23 Fix: Place at E2
        self.place_at_grid(byte, "E2", scale_factor=0.8)
        
        # Basketball Hoop
        hoop_pole = Line(self.grid["F5"], self.grid["D5"], color=WHITE)
        hoop_backboard = Rectangle(height=0.6, width=0.1, color=WHITE).next_to(hoop_pole.get_top(), LEFT, buff=0)
        hoop_rim = Ellipse(width=0.5, height=0.15, color="#FFFFFF").next_to(hoop_backboard, LEFT, buff=0).shift(DOWN*0.1)
        hoop = VGroup(hoop_pole, hoop_backboard, hoop_rim)
        
        self.play(FadeIn(byte), FadeIn(hoop))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Issue 20 Fix: Use Basketball Asset
        ball = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/basketball.svg")
        ball.set_color("#FFA500")
        ball.scale(0.3)
        ball.move_to(byte.get_top() + RIGHT * 0.3)
        
        # Paths
        start_point = ball.get_center()
        success_end = hoop_rim.get_center()
        failure_end = hoop_rim.get_center() + DOWN*1.5 + RIGHT*0.5
        
        path_p = CurvedArrow(start_point, success_end, angle=-TAU/4, color="#00FF00")
        path_q = CurvedArrow(start_point, failure_end, angle=-TAU/6, color="#FF0000")
        
        # Using Text with slant=ITALIC
        label_p = Text("p", slant=ITALIC, color="#00FF00", font_size=30).next_to(path_p, UP, buff=0.1)
        label_q = Text("1-p", slant=ITALIC, color="#FF0000", font_size=30).next_to(path_q, DOWN, buff=0.1)

        # Animate the paths and labels
        self.play(Create(path_p), FadeIn(label_p))
        self.play(Create(path_q), FadeIn(label_q))
        
        # Shot animation (Success)
        self.play(FadeIn(ball))
        self.play(MoveAlongPath(ball, path_p), run_time=1.5)
        self.play(ball.animate.shift(DOWN * 0.5), rate_func=there_and_back)
        self.wait(0.5)
        self.play(FadeOut(ball))
        
        # Reset ball for second trial visual (Failure)
        ball_fail = ball.copy()
        ball_fail.move_to(start_point)
        self.play(FadeIn(ball_fail))
        self.play(MoveAlongPath(ball_fail, path_q), run_time=1.5)
        self.wait(1)

        # End of section
        self.wait(2)
