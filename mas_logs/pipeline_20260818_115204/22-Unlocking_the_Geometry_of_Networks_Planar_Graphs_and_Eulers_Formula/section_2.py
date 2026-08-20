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
        self.setup_layout("Core Components: Vertices, Edges, and Faces", 
                          ["Graphs consist of vertices and edges.", 
                           "Faces are the enclosed regions.", 
                           "The infinite area is a face."])
        
        # Elements
        v = Dot(color="#00FF00")
        v_label = Text("V", color="#00FF00", font_size=24)
        v_group = VGroup(v, v_label).arrange(UP, buff=0.1)
        
        e = Line(start=np.array([0, 0, 0]), end=np.array([1, 0, 0]), color="#0000FF")
        e_label = Text("E", color="#0000FF", font_size=24)
        e_group = VGroup(e, e_label).arrange(UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(v_group, 'B2', scale_factor=0.8)
        self.play(Create(v), Write(v_label))
        
        # Adding edge after vertex
        self.place_at_grid(e_group, 'B4', scale_factor=0.8)
        self.play(Create(e), Write(e_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Create a simple triangle to represent a face
        face = Polygon(np.array([0, 0, 0]), np.array([0.5, 0.8, 0]), np.array([1, 0, 0]), color="#FF00FF", fill_opacity=0.3)
        self.place_at_grid(face, 'D3', scale_factor=0.9)
        self.play(DrawBorderThenFill(face))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        infinite_box = Rectangle(color="#FFFF00", width=3, height=3)
        self.place_in_area(infinite_box, 'C2', 'F6', scale_factor=0.9)
        self.play(Create(infinite_box))
        self.wait(1)
