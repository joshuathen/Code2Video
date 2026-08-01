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
        # Configuration
        TITLE = "Bernoulli Trial: Success or Failure"
        LINES = [
            "Bernoulli trials are the building blocks of binomial distributions.",
            "Meet Finn. Every fly catch is a single trial.",
            "Success means he catches the fly.",
            "Failure means the fly escapes.",
            "Finn's success rate stays constant at seventy percent."
        ]
        FINN_COLOR = "#00FF00"
        FLY_COLOR = "#A9A9A9"
        SUCCESS_COLOR = "#FFD700"
        FAILURE_COLOR = "#A9A9A9"
        TEXT_COLOR = "#FFFFFF"

        self.setup_layout(TITLE, LINES)
        
        # Initially hide elements to control entrance timing
        self.remove(self.title, self.lecture)

        # === Animation for Lecture Line 1 ===
        # [self.wait(2.0)] Fade in title 'Bernoulli Trial: Success or Failure' in #FFFFFF.
        self.wait(2.0)
        self.play(FadeIn(self.title), FadeIn(self.lecture))
        self.play(self.lecture[0].animate.set_color(SUCCESS_COLOR))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Meet Finn. Every fly catch is a single trial.
        self.play(
            self.lecture[0].animate.set_color(TEXT_COLOR),
            self.lecture[1].animate.set_color(FINN_COLOR)
        )
        self.wait(1.5)
        
        # Finn the Frog (Hand-built primitive)
        finn_body = Ellipse(width=1.2, height=0.8, color=FINN_COLOR, fill_opacity=1)
        eye_l = Dot(color="#000000").shift(UP*0.2 + LEFT*0.3)
        eye_r = Dot(color="#000000").shift(UP*0.2 + RIGHT*0.3)
        finn = VGroup(finn_body, eye_l, eye_r)
        self.place_at_grid(finn, "D2", scale_factor=0.8)
        
        # The Fly (Hand-built primitive)
        fly_body = Dot(radius=0.1, color="#333333")
        wing_l = Ellipse(width=0.2, height=0.1, color="#888888", fill_opacity=0.5).shift(LEFT*0.1 + UP*0.05)
        wing_r = Ellipse(width=0.2, height=0.1, color="#888888", fill_opacity=0.5).shift(RIGHT*0.1 + UP*0.05)
        fly = VGroup(fly_body, wing_l, wing_r)
        self.place_at_grid(fly, "B5", scale_factor=0.8)

        self.play(FadeIn(finn), FadeIn(fly))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Success means he catches the fly.
        self.play(
            self.lecture[1].animate.set_color(TEXT_COLOR),
            self.lecture[2].animate.set_color(SUCCESS_COLOR)
        )
        self.wait(1.5)
        
        success_label = Text("SUCCESS", color=SUCCESS_COLOR, font_size=24)
        self.place_at_grid(success_label, "A5", scale_factor=0.8)

        # Catch animation: Tongue shoots out and retracts with fly
        tongue = Line(finn.get_top(), fly.get_center(), color="#FF69B4", stroke_width=4)
        self.play(Create(tongue), run_time=0.3)
        self.play(fly.animate.move_to(finn.get_top()), run_time=0.3)
        self.play(FadeOut(tongue), FadeOut(fly), Write(success_label))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Failure means the fly escapes.
        self.play(
            self.lecture[2].animate.set_color(TEXT_COLOR),
            self.lecture[3].animate.set_color(FAILURE_COLOR)
        )
        self.wait(1.5)
        
        # New Fly instance for the second trial
        fly_body_2 = Dot(radius=0.1, color="#333333")
        wing_l_2 = Ellipse(width=0.2, height=0.1, color="#888888", fill_opacity=0.5).shift(LEFT*0.1 + UP*0.05)
        wing_r_2 = Ellipse(width=0.2, height=0.1, color="#888888", fill_opacity=0.5).shift(RIGHT*0.1 + UP*0.05)
        fly_2 = VGroup(fly_body_2, wing_l_2, wing_r_2)
        self.place_at_grid(fly_2, "B5", scale_factor=0.8)
        
        failure_label = Text("FAILURE", color=FAILURE_COLOR, font_size=24)
        # Corrected position for failure_label per Issue 21
        self.place_at_grid(failure_label, "A5", scale_factor=0.8)

        self.play(FadeOut(success_label), FadeIn(fly_2))
        
        # Escape animation: Fly moves quickly away
        self.play(fly_2.animate.move_to(self.grid["A6"]), run_time=1)
        self.play(Write(failure_label))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Finn's success rate stays constant at seventy percent.
        self.play(
            self.lecture[3].animate.set_color(TEXT_COLOR),
            self.lecture[4].animate.set_color(SUCCESS_COLOR)
        )
        self.wait(2.0)
        
        # Using MathTex for probability representation
        prob_text = MathTex("p = 0.70", color=SUCCESS_COLOR)
        # Corrected position for prob_text per Issue 20
        self.place_at_grid(prob_text, "D3", scale_factor=1.0)
        rect = SurroundingRectangle(prob_text, color=SUCCESS_COLOR, buff=0.2)
        
        self.play(Write(prob_text))
        self.play(Create(rect))
        self.play(Indicate(prob_text, color=SUCCESS_COLOR))
        self.wait(2.0)
