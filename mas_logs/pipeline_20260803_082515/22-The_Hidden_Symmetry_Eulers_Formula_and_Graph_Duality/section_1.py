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
        title = "Prerequisite: Anatomy of a Planar Graph"
        lines = [
            "Planar graphs are drawn without any edges crossing.",
            "Vertices are points, while edges are connecting lines.",
            "Faces are regions bounded by edges, including the outside."
        ]
        self.setup_layout(title, lines)
        
        # Colors for matching lecture lines
        COLOR_L1 = WHITE
        COLOR_L2 = "#FFFFE0" # Light Yellow for vertices
        COLOR_E  = "#ADD8E6" # Light Blue for edges
        COLOR_F1 = "#ADD8E6" # Light Blue for Face 1
        COLOR_F2 = "#90EE90" # Light Green for Face 2
        
        # === Animation for Lecture Line 1 ===
        # Planar graphs are drawn without any edges crossing.
        self.play(self.lecture[0].animate.set_color(COLOR_L1))
        
        # Load Square Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg]
        square_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        square_asset.set_color(COLOR_E)
        self.place_in_area(square_asset, "B2", "E5", scale_factor=1.5)
        
        # Define vertices and edges explicitly for highlighting later
        v1 = Dot(self.grid["B2"], color=WHITE)
        v2 = Dot(self.grid["B5"], color=WHITE)
        v3 = Dot(self.grid["E5"], color=WHITE)
        v4 = Dot(self.grid["E2"], color=WHITE)
        
        e1 = Line(v1.get_center(), v2.get_center(), color=COLOR_E)
        e2 = Line(v2.get_center(), v3.get_center(), color=COLOR_E)
        e3 = Line(v3.get_center(), v4.get_center(), color=COLOR_E)
        e4 = Line(v4.get_center(), v1.get_center(), color=COLOR_E)
        
        graph_v = VGroup(v1, v2, v3, v4)
        graph_e = VGroup(e1, e2, e3, e4)
        
        # Animation: Create the graph (using manual components to ensure control)
        self.play(Create(graph_e), Create(graph_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vertices are points, while edges are connecting lines.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_L2)
        )
        
        # Highlight vertices and edges
        v_label = Text("Vertices", font_size=18, color=COLOR_L2)
        e_label = Text("Edges", font_size=18, color=COLOR_E)
        
        self.place_at_grid(v_label, "A2")
        # Issue 20 Fix: e_label at C6
        self.place_at_grid(e_label, "C6", scale_factor=0.8)
        
        self.play(
            graph_v.animate.set_color(COLOR_L2),
            Write(v_label),
            Write(e_label)
        )
        self.play(Indicate(graph_v), Indicate(graph_e))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Faces are regions bounded by edges, including the outside.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_F1)
        )
        
        # Face 1 (Inner)
        face1 = Polygon(
            v1.get_center(), v2.get_center(), v3.get_center(), v4.get_center(), 
            fill_opacity=0.5, fill_color=COLOR_F1, stroke_width=0
        )
        f1_label = Text("Face 1 (Inner)", font_size=18, color=COLOR_F1)
        # Issue 21 Fix: f1_label in area C3-D4
        self.place_in_area(f1_label, "C3", "D4", scale_factor=0.7)
        
        self.play(FadeIn(face1), Write(f1_label))
        self.wait(1)
        
        # Face 2 (Outer)
        # Use a rectangle for the background area to represent the outer face
        face2_bg = Rectangle(
            width=6.0, height=5.5, fill_opacity=0.2, fill_color=COLOR_F2, stroke_width=0
        )
        self.place_in_area(face2_bg, "A1", "F6")
        
        f2_label = Text("Face 2 (Outer)", font_size=18, color=COLOR_F2)
        # Issue 22 Fix: f2_label in area F2-F5
        self.place_in_area(f2_label, "F2", "F5", scale_factor=0.8)
        
        # Ensure correct layering
        self.play(FadeIn(face2_bg), Write(f2_label))
        self.add(face1, f1_label, graph_v, graph_e, v_label, e_label)
        
        self.wait(2)
