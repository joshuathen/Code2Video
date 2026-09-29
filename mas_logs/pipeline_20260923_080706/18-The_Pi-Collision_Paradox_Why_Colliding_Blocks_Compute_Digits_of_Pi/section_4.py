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
            "Map 1D motion to 2D.",
            "Use a wedge shaped boundary.",
            "Path traces a circular arc.",
            "Geometry calculates digits of Pi.",
            "Unfolding reveals the circular logic."
        ]
        self.setup_layout("Geometric Visualization: Unfolding", lecture_lines)
        
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Setup visualization objects
        angle = 30 * DEGREES
        
        # 1. Wedge (Fix for issue 30)
        wedge = Sector(radius=1.5, start_angle=0, angle=angle, color=WHITE, fill_opacity=0.2)
        self.place_in_area(wedge, 'B3', 'C4', scale_factor=1.2)
        
        # 2. Path (Fix for issue 31)
        arc_path = Arc(radius=1.5, start_angle=0, angle=angle, color=PURPLE)
        self.place_in_area(arc_path, 'B3', 'D5', scale_factor=1.0)
        
        # 3. Label (Fix for issue 32)
        geometry_label = Text("2D Wedge Projection", font_size=20, color=WHITE)
        self.place_at_grid(geometry_label, 'D3', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.play(FadeIn(wedge))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        self.play(Create(wedge))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        self.play(Create(arc_path), FadeIn(geometry_label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.play(Rotate(arc_path, angle=0.1, about_point=self.grid['C3']))
        self.wait(2)
