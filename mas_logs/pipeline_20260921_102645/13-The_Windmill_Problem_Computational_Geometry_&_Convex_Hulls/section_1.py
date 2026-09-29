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
        self.setup_layout("Introduction: The Setting", [
            "Imagine points scattered across a plane.", 
            "A line pivots around one point.", 
            "It sweeps until it hits another.", 
            "This process repeats indefinitely.", 
            "This creates the windmill motion."
        ])
        
        # Points
        points_data = ['B2', 'B5', 'C4', 'D2', 'E4']
        points = VGroup(*[Dot(self.grid[pos], color=WHITE) for pos in points_data])
        
        # Windmill asset
        windmill = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg", color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00CED1")
        self.play(FadeIn(points))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        pivot = points[0]
        # Position windmill at pivot
        self.place_at_grid(windmill, 'B2', scale_factor=0.3)
        self.play(FadeIn(windmill))
        # Fix 21: use place_in_area for the line object
        line = Line(start=pivot.get_center() + LEFT*1.5, end=pivot.get_center() + RIGHT*1.5, color=BLUE)
        self.place_in_area(line, 'C4', 'F6', scale_factor=0.9)
        self.play(Create(line))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF6347")
        # Rotating until it hits points[1]
        target_angle = angle_between_vectors(line.get_vector(), points[1].get_center() - pivot.get_center())
        # Fix 22: Hit Point Label
        hit_label = Text("Hit!", font_size=18, color=YELLOW)
        self.place_at_grid(hit_label, 'D3', scale_factor=0.6)
        
        self.play(
            Rotate(line, angle=target_angle, about_point=pivot.get_center()),
            Rotate(windmill, angle=target_angle, about_point=pivot.get_center()),
            Write(hit_label)
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#7FFF00")
        # Highlight points
        self.play(Indicate(points[0]), Indicate(points[1]))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        # Shift pivot to point[1]
        new_pivot = points[1]
        # Update windmill position
        windmill.move_to(new_pivot.get_center())
        self.wait(1)
