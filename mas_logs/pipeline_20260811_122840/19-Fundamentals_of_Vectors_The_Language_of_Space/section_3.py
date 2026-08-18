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
        self.setup_layout("The Components of a Vector", ["Break vectors into x and y.", "Use i and j basis units.", "Combine them to form vectors."])
        
        # Asset grid
        grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid, 'C3', 'F6', scale_factor=0.7)
        self.add(grid)

        # Define vector
        origin = self.grid["F4"]
        end = self.grid["C6"]
        vec_v = Arrow(origin, end, color="#FFFFFF", buff=0)
        label_v = Text("v", font_size=20, color="#FFFFFF")
        label_v.next_to(vec_v.get_center(), UP, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(vec_v), Write(label_v))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Components
        vx = Line(origin, [end[0], origin[1], 0], color="#FF0000", stroke_width=4)
        vy = Line([end[0], origin[1], 0], end, color="#00FF00", stroke_width=4)
        
        label_vx = Text("vx", font_size=20, color="#FF0000")
        self.place_at_grid(label_vx, 'D4', scale_factor=0.9)
        
        label_vy = Text("vy", font_size=20, color="#00FF00")
        self.place_at_grid(label_vy, 'E5', scale_factor=0.9)
        
        self.play(Create(vx), Create(vy), Write(label_vx), Write(label_vy))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(Indicate(vec_v))
