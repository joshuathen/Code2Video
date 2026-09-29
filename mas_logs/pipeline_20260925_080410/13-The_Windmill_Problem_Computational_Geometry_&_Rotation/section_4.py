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
        lecture_lines = [
            "First, select an initial pivot point.",
            "Calculate angles to all other points.",
            "Rotate the line to the smallest angle.",
            "Update the pivot to the hit point.",
            "Repeat this process for the full set."
        ]
        self.setup_layout("Visualization of the Rotation", lecture_lines)
        
        # Load Windmill asset
        windmill = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        self.place_in_area(windmill, 'A4', 'F6', scale_factor=0.5)
        
        # Setup points
        points_pos = ["B4", "B6", "E4", "E6", "C5"]
        points = [Dot(self.grid[pos], color=BLUE) for pos in points_pos]
        points_group = VGroup(*points)
        self.add(points_group)
        
        # Rotation Line
        line = Line(start=points[0].get_center(), end=points[4].get_center(), color=YELLOW)
        self.add(line)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(FadeIn(windmill))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        angles = [DashedLine(points[0].get_center(), p.get_center(), color=GRAY) for p in points[1:]]
        self.play(*[Create(a) for a in angles], run_time=1)
        self.wait(1)
        self.play(*[FadeOut(a) for a in angles])

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        pivot = points[0]
        # Trace path
        path = Line(line.get_start(), line.get_end(), color="#FF00FF")
        self.add(path)
        self.play(Rotate(line, angle=PI/6, about_point=pivot.get_center()), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        target = points[1]
        self.play(line.animate.put_start_and_end_on(target.get_center(), pivot.get_center()), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.play(Flash(target.get_center()), run_time=1)
        self.wait(2)
