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
        self.setup_layout("Prerequisite: Entropy and Information Gain", [
            "Entropy measures our total uncertainty.", 
            "High entropy equals high information gain.", 
            "Optimal guesses partition words evenly."
        ])
        
        # Colors for lecture lines
        c1, c2, c3 = "#FFFFFF", "#FFFF00", "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Show text 'Entropy Basics' in color #FFFFFF
        text_label = Text("Entropy Basics", font_size=32, color="#FFFFFF")
        self.place_at_grid(text_label, 'C3', scale_factor=0.9)
        self.play(FadeIn(text_label))
        self.lecture[0].set_color(c1)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Display bar chart for probability distribution #00FF00
        bars = VGroup(*[Rectangle(height=1.5 * (i+1)/5, width=0.5, color="#00FF00", fill_opacity=0.7) for i in range(5)])
        bars.arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
        self.place_in_area(bars, 'D2', 'F5', scale_factor=1.1)
        self.play(Create(bars))
        self.lecture[1].set_color(c2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show the space partitioning into equal buckets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg] in #FF0000, color line 3 in #FFFF00.
        bucket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg")
        bucket.set_color("#FF0000")
        self.place_at_grid(bucket, 'B5', scale_factor=0.6)
        
        bucket_label = Text("Bucket", font_size=24, color="#FF0000")
        self.place_at_grid(bucket_label, 'C5', scale_factor=0.7)
        
        self.play(FadeIn(bucket), Write(bucket_label))
        self.lecture[2].set_color(c3)
        self.wait(2)
