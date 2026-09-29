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
        self.setup_layout("The Anatomy of a Measure (Time Signatures)", [
            "Time signatures look like fractions in math.",
            "The top number counts beats per measure.",
            "The bottom number defines the note value."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a simple 4/4 time signature fraction using Asset. (Color: #FFFFFF)
        time_sig = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fraction.svg", color=WHITE)
        self.place_at_grid(time_sig, 'B3', scale_factor=1.2)
        self.play(FadeIn(time_sig))
        self.wait(4)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Partition a circle into 4 equal rhythmic segments. (Color: #00FF00)
        circle = Circle(radius=1.5, color=WHITE)
        lines = VGroup(*[Line(ORIGIN, 1.5*RIGHT).rotate(i * PI/2, about_point=ORIGIN) for i in range(4)])
        pie_chart = VGroup(circle, lines)
        self.place_at_grid(pie_chart, 'D5', scale_factor=0.6)
        
        self.play(Create(circle), Create(lines))
        for line in lines:
            line.set_color("#00FF00")
        self.wait(4)
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Highlight the first beat as the downbeat. (Color: #FF0000)
        # Moving to F5 for separation as requested
        downbeat = Sector(radius=1.5, start_angle=0, angle=PI/2, color="#FF0000", fill_opacity=0.5)
        self.place_at_grid(downbeat, 'F5', scale_factor=0.6)
        
        self.play(FadeIn(downbeat))
        self.wait(4)
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
