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
        # Fetching data for section_1
        title_text = "The Mystery of Chaos"
        lecture_lines = [
            "The world is often messy and unpredictable.",
            "Individual data points can appear completely random.",
            "Can we find order within this chaotic data?"
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Visual styling
        self.title.set_color("#ADD8E6") # Light blue as per storyboard
        YELLOW_HEX = "#FFFF00"
        RED_HEX = "#FF0000"
        GREEN_HEX = "#00FF00"
        WHITE_HEX = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display a chaotic scatter plot of dots with random sizes in yellow.
        self.play(self.lecture[0].animate.set_color(YELLOW_HEX))
        
        dots = VGroup()
        np.random.seed(42)
        # Use grid area B1 to E6 for the chaotic dots
        tl = self.grid["B1"]
        br = self.grid["E6"]
        
        for _ in range(40):
            rand_x = np.random.uniform(tl[0], br[0])
            rand_y = np.random.uniform(br[1], tl[1])
            dot = Dot(
                point=[rand_x, rand_y, 0], 
                radius=np.random.uniform(0.04, 0.12), 
                color=YELLOW_HEX
            )
            dots.add(dot)
            
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash extreme dots in red to show data variability.
        self.play(self.lecture[1].animate.set_color(RED_HEX))
        
        # Select the most "extreme" dots based on their size
        sorted_dots = sorted(dots, key=lambda d: d.radius)
        extreme_dots = VGroup(sorted_dots[0], sorted_dots[1], sorted_dots[-1], sorted_dots[-2])
        
        # Flashing red to highlight variability
        self.play(*[Flash(d, color=RED_HEX, flash_radius=0.3) for d in extreme_dots])
        self.play(extreme_dots.animate.set_color(RED_HEX), run_time=0.5)
        self.wait(1)
        self.play(extreme_dots.animate.set_color(YELLOW_HEX), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        # Show a white pulsing question mark in the center of the chaos.
        # Transition dots into a skewed green population bar chart.
        self.play(self.lecture[2].animate.set_color(GREEN_HEX))
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/question.svg
        question_mark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/question.svg").set_color(WHITE_HEX)
        self.place_in_area(question_mark, "B3", "E4") # Fix for Issue 23
        
        self.play(FadeIn(question_mark))
        self.play(question_mark.animate.scale(1.2), rate_func=there_and_back, run_time=0.8)
        self.wait(0.5)
        
        # Create a skewed green bar chart
        bar_heights = [0.5, 1.2, 3.2, 2.0, 1.0, 0.6]
        bars = VGroup(*[
            Rectangle(
                height=h, 
                width=0.6, 
                fill_opacity=0.8, 
                fill_color=GREEN_HEX, 
                stroke_color=GREEN_HEX
            ) for h in bar_heights
        ]).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
        
        self.place_in_area(bars, "B2", "E5") # Fix for Issue 24
        
        # Organize dots into groups to transform into bars smoothly
        dot_groups = [VGroup() for _ in range(len(bars))]
        for i, dot in enumerate(dots):
            dot_groups[i % len(bars)].add(dot)
            
        self.play(
            FadeOut(question_mark),
            *[ReplacementTransform(dg, bar) for dg, bar in zip(dot_groups, bars)]
        )
        self.wait(3)
