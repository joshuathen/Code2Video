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
        lecture_lines = [
            "Discrete variables like coin flips have countable outcomes.",
            "Continuous variables like time have infinite possibilities.",
            "Meet Pip. Can we measure a lightbulb's exact life?",
            "Probability of lasting exactly 100.000 hours is zero.",
            "In continuous ranges, individual points have no probability."
        ]
        self.setup_layout("The Shift: From Counting to Measuring", lecture_lines)

        # === Robot "Pip" Construction ===
        # Using the provided asset for Pip
        pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(pip, "B2", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Silver coin setup
        coin = Circle(radius=0.5, color="#C0C0C0", fill_opacity=1)
        coin_text = Text("H", color=BLACK, font_size=36)
        coin_group = VGroup(coin, coin_text)
        self.place_in_area(coin_group, "B4", "B6", scale_factor=0.8)
        
        self.play(FadeIn(pip), FadeIn(coin_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEAL)
        
        # Coin flip animation (discrete)
        self.play(Rotate(coin_group, angle=PI*2, axis=RIGHT, run_time=1))
        # Update text manually
        new_coin_text = Text("T", color=BLACK, font_size=36).move_to(coin.get_center())
        coin_text.become(new_coin_text)
        self.play(Flash(coin_group, color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Lightbulb Construction
        bulb_glass = Circle(radius=0.5, color="#FFFFE0", fill_opacity=0.8)
        bulb_base = Rectangle(width=0.3, height=0.2, color=GRAY, fill_opacity=1).next_to(bulb_glass, DOWN, buff=0)
        bulb_filament = VGroup(
            Line(LEFT*0.1, RIGHT*0.1, color=YELLOW),
            Line(ORIGIN, UP*0.2, color=YELLOW)
        ).move_to(bulb_glass.get_center())
        lightbulb = VGroup(bulb_glass, bulb_base, bulb_filament)
        # Replacing the coin group area
        self.place_in_area(lightbulb, "B4", "B6", scale_factor=0.8)

        self.play(ReplacementTransform(coin_group, lightbulb))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # Timer / Measurements
        timer_label = Text("Time:", font_size=20, color=WHITE)
        timer_val = Text("100.125", font_size=20, color=YELLOW)
        timer_group = VGroup(timer_label, timer_val).arrange(RIGHT, buff=0.2)
        self.place_in_area(timer_group, "C4", "C6", scale_factor=0.7)
        
        self.play(FadeIn(timer_group))
        
        # Counting up keyframes
        timer_val_next1 = Text("100.126", font_size=20, color=YELLOW).move_to(timer_val)
        self.play(Transform(timer_val, timer_val_next1), run_time=0.5)
        
        timer_val_next2 = Text("100.127", font_size=20, color=YELLOW).move_to(timer_val)
        self.play(Transform(timer_val, timer_val_next2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(ORANGE)
        
        # Timeline
        timeline = Line(start=self.grid["E1"], end=self.grid["E6"], color=WHITE)
        ticks = VGroup(*[Line(timeline.point_from_proportion(i/4) + UP*0.1, timeline.point_from_proportion(i/4) + DOWN*0.1, color=WHITE) for i in range(5)])
        labels = VGroup(
            Text("99.9", font_size=16).next_to(ticks[1], DOWN),
            Text("100.0", font_size=16).next_to(ticks[2], DOWN),
            Text("100.1", font_size=16).next_to(ticks[3], DOWN)
        )
        timeline_group = VGroup(timeline, ticks, labels)
        
        # Point and "Zero Probability" Markings
        point_mark = Dot(ticks[2].get_center(), color=RED, radius=0.05)
        zero_p = Text("P = 0", color=RED, font_size=18).next_to(point_mark, UP, buff=0.1)
        red_x = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color="#FF0000"),
            Line(UP+RIGHT, DOWN+LEFT, color="#FF0000")
        ).scale(0.15).move_to(point_mark.get_center())

        self.play(FadeOut(lightbulb), FadeOut(timer_group))
        self.play(Create(timeline_group))
        self.play(Create(point_mark), Write(zero_p))
        self.play(Create(red_x))
        
        self.wait(2)
