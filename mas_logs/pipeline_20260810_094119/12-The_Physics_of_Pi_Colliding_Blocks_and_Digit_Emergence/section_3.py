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
        lecture_lines = ["Map velocity states onto a circle.", "Each collision rotates the velocity vector.", "Total rotation matches Pi's geometry.", "Collision count relates to arc length.", "Geometric mapping explains this Pi connection."]
        self.setup_layout("Geometric Mapping: The Circle Connection", lecture_lines)
        
        # Load Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Elements
        circle_container = VGroup(
            Circle(radius=1.0, color=WHITE),
            compass
        )
        point = Dot(color="#FFFF00")
        projection = DashedLine(start=ORIGIN, end=RIGHT, color="#A9A9A9")
        arc_label = Text("Pi", color="#FFD700")
        
        # Position using grid
        self.place_in_area(circle_container, 'A4', 'C6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(circle_container))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(FadeIn(point))
        self.play(Rotate(point, angle=PI/4, about_point=circle_container[0].get_center(), run_time=2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#A9A9A9")
        self.play(Create(projection))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.play(Rotate(point, angle=PI/4, about_point=circle_container[0].get_center(), run_time=2))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        self.place_in_area(ruler, 'D4', 'D5', scale_factor=0.8)
        self.play(FadeIn(ruler), Write(arc_label.next_to(ruler, DOWN)))
        self.wait(1)
