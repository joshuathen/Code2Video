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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Notes are like fractions of time.", "Whole notes fill the entire measure.", "Smaller notes divide the circle equally.", "They sum perfectly to one measure.", "Each wedge completes our musical clock."]
        self.setup_layout("Visualizing Fractions in Time", lecture_lines)
        
        # Elements
        # Using SVGMobject for clock as requested
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        clock.set_color(WHITE)
        self.place_at_grid(clock, 'B4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(clock), run_time=1)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(PINK)
        # Whole note = clock itself
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        wedge1 = Sector(start_angle=PI/2, angle=PI, color="#FF00FF", fill_opacity=0.6)
        wedge2 = Sector(start_angle=3*PI/2, angle=PI, color="#FF00FF", fill_opacity=0.6)
        self.place_at_grid(wedge1, 'B4', scale_factor=0.7)
        self.place_at_grid(wedge2, 'B4', scale_factor=0.7)
        self.play(FadeIn(wedge1), FadeIn(wedge2))
        self.wait(3)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        label1 = Text("1/2", font_size=24, color=WHITE)
        label2 = Text("1/2", font_size=24, color=WHITE)
        label1.move_to(clock.get_center() + UP*0.3)
        label2.move_to(clock.get_center() + DOWN*0.3)
        self.play(FadeOut(wedge1), FadeOut(wedge2), Write(label1), Write(label2))
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED)
        # Clock color change
        clock.set_color(ManimColor.from_hex("#FF0000")) # Dummy start
        self.play(clock.animate.set_color(ManimColor.from_hex("#00FF00")), run_time=2)
        
        flash = Circle(radius=0.8, color=YELLOW, stroke_width=4)
        self.place_at_grid(flash, 'B4', scale_factor=0.9)
        self.play(Create(flash), run_time=0.5)
        self.play(Flash(clock.get_center(), color=YELLOW, line_length=0.2, num_lines=12))
        self.play(FadeOut(flash))
        self.wait(3)
