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
        # Data from storyboard
        title_text = "Prerequisite: Independence vs. Dependence"
        lecture_lines = [
            "Independent events do not influence each other's probabilities.",
            "Coin tosses are independent; drawing marbles may not be.",
            "Cloudy weather and rain are clearly dependent events."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display a split-screen: "Independent" (#00FF00) vs "Dependent" (#FF0000).
        self.lecture[0].set_color("#00FF00")
        
        independent_label = Text("Independent", color="#00FF00", font_size=24)
        dependent_label = Text("Dependent", color="#FF0000", font_size=24)
        
        self.place_in_area(independent_label, "A1", "A3")
        self.place_in_area(dependent_label, "A4", "A6")
        
        # Split line between column 3 and 4
        x_split = (self.grid["A3"][0] + self.grid["A4"][0]) / 2
        split_line = Line(
            start=[x_split, 2.5, 0],
            end=[x_split, -3.5, 0],
            color=GRAY,
            stroke_width=2
        )
        
        self.play(
            Write(independent_label),
            Write(dependent_label),
            Create(split_line),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # On the left, animate a die [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg] (#FFFFFF)
        # and a coin [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg] (#FFD700) flip simultaneously.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700") # Highlight color
        
        # Die at B1, Label B2 (Issue 33)
        die = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg").set_color(WHITE)
        self.place_at_grid(die, "B1", scale_factor=0.6)
        
        die_label = Text("Die Roll", font_size=18, color=WHITE)
        self.place_at_grid(die_label, "B2")
        
        # Coin at D1, Label D2 (Issue 33)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg").set_color("#FFD700")
        self.place_at_grid(coin, "D1", scale_factor=0.6)
        
        coin_label = Text("Coin Flip", font_size=18, color="#FFD700")
        self.place_at_grid(coin_label, "D2")
        
        self.play(
            FadeIn(die), Write(die_label),
            FadeIn(coin), Write(coin_label),
            run_time=1
        )
        
        # Simulating roll and flip
        self.play(
            die.animate.rotate(PI),
            coin.animate.rotate(PI, axis=Y_AXIS),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # On the right, animate a cloud (#888888) appearing followed by rain (#0000FF), linked by a glowing arrow.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#0000FF") # Highlight color
        
        # Cloud at B4, Label B5 (Issue 34)
        cloud = VGroup(
            Circle(radius=0.2, color="#888888", fill_opacity=0.8).shift(LEFT*0.2),
            Circle(radius=0.3, color="#888888", fill_opacity=0.8).shift(UP*0.1),
            Circle(radius=0.2, color="#888888", fill_opacity=0.8).shift(RIGHT*0.2)
        )
        self.place_at_grid(cloud, "B4", scale_factor=0.8)
        cloud_label = Text("Cloudy", font_size=18, color="#888888")
        self.place_at_grid(cloud_label, "B5")
        
        # Rain at D4, Label D5 (Issue 34)
        rain_drops = VGroup(*[
            Line(start=ORIGIN, end=DOWN*0.15, color="#0000FF", stroke_width=2).shift(RIGHT*dx + UP*dy)
            for dx, dy in [(-0.15,0), (0, -0.1), (0.15,0), (-0.05, -0.2), (0.1, -0.2)]
        ])
        self.place_at_grid(rain_drops, "D4", scale_factor=0.8)
        rain_label = Text("Rain", font_size=18, color="#0000FF")
        self.place_at_grid(rain_label, "D5")
        
        # Glowing Arrow (from Cloud area to Rain area)
        arrow_start = self.grid["B4"] + DOWN * 0.4
        arrow_end = self.grid["D4"] + UP * 0.4
        arrow = Arrow(start=arrow_start, end=arrow_end, color=WHITE, buff=0.1, stroke_width=4)
        glow = arrow.copy().set_stroke(color=WHITE, width=12, opacity=0.2)
        
        self.play(FadeIn(cloud), Write(cloud_label), run_time=1)
        self.play(Create(glow), Create(arrow), run_time=0.8)
        self.play(FadeIn(rain_drops), Write(rain_label), run_time=1)
        
        # Brief rain motion
        self.play(
            rain_drops.animate.shift(DOWN*0.2).set_opacity(0),
            rate_func=linear,
            run_time=0.6
        )
        
        self.wait(2)
