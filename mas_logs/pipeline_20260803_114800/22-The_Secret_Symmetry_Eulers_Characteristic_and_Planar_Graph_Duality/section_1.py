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
        self.setup_layout(
            "Prerequisites: Anatomy of a Planar Graph", 
            [
                "Planar graphs are drawn without crossing edges.",
                "They consist of vertices, edges, and faces.",
                "Every graph includes an infinite outer face."
            ]
        )
        
        # Colors
        v_color = "#00FFFF"
        e_color = "#FFFFFF"
        f_color = "#FFFF00"

        # Asset integration (Issue 16)
        # We use the SVG as the base visual for the house-shaped graph.
        house_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg")
        self.place_in_area(house_asset, "B2", "E4", scale_factor=1.5)
        house_asset.set_color(e_color)

        # Define Vertices (Placed at grid junctions to form the house shape)
        v_peak = Dot(color=v_color).move_to(self.grid["B3"])
        v_tl = Dot(color=v_color).move_to(self.grid["C2"])
        v_tr = Dot(color=v_color).move_to(self.grid["C4"])
        v_bl = Dot(color=v_color).move_to(self.grid["E2"])
        v_br = Dot(color=v_color).move_to(self.grid["E4"])
        
        vertices = VGroup(v_bl, v_br, v_tr, v_tl, v_peak)

        # Define Edges (Lines connecting the vertices)
        edges = VGroup(
            Line(v_bl.get_center(), v_br.get_center(), color=e_color), # Floor
            Line(v_br.get_center(), v_tr.get_center(), color=e_color), # Right wall
            Line(v_tr.get_center(), v_tl.get_center(), color=e_color), # Ceiling
            Line(v_tl.get_center(), v_bl.get_center(), color=e_color), # Left wall
            Line(v_tl.get_center(), v_peak.get_center(), color=e_color), # Left roof
            Line(v_tr.get_center(), v_peak.get_center(), color=e_color)  # Right roof
        )

        # Faces (Polygons for highlighting)
        # Inner Face 1: The square body of the house
        f1_poly = Polygon(
            v_bl.get_center(), v_br.get_center(), v_tr.get_center(), v_tl.get_center(),
            fill_color=f_color, fill_opacity=0.4, stroke_width=0
        )
        # Inner Face 2: The triangle roof
        f2_poly = Polygon(
            v_tl.get_center(), v_tr.get_center(), v_peak.get_center(),
            fill_color=f_color, fill_opacity=0.4, stroke_width=0
        )
        # Outer Face: Represented as a surrounding rectangle
        outer_face_box = SurroundingRectangle(edges, color=f_color, buff=0.8, fill_opacity=0.1, fill_color=f_color)

        # === Animation for Lecture Line 1 ===
        # Planar graphs are drawn without crossing edges.
        self.play(self.lecture[0].animate.set_color(v_color))
        self.play(
            Create(house_asset),
            Create(edges),
            Create(vertices),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They consist of vertices, edges, and faces.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(f_color)
        )
        
        # Highlight inner faces one by one
        self.play(FadeIn(f1_poly))
        self.wait(0.5)
        self.play(FadeIn(f2_poly))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every graph includes an infinite outer face.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(f_color)
        )
        
        # Highlight outer face
        self.play(Create(outer_face_box))
        self.wait(1)

        # Final Labels (Addressing Issues 18, 19, 20)
        v_label = Text("V = 5", font_size=24, color=WHITE)
        e_label = Text("E = 6", font_size=24, color=WHITE)
        f_label = Text("F = 3", font_size=24, color=WHITE)
        
        # Positioning labels at the bottom row (F) with updated column positions
        self.place_at_grid(v_label, 'F3') # Fix for Issue 18: Move from F2 to F3
        self.place_at_grid(e_label, 'F4') # Fix for Issue 19: Move from F3 to F4
        self.place_at_grid(f_label, 'F5') # Fix for Issue 20: Move from F4 to F5

        self.play(
            Write(v_label),
            Write(e_label),
            Write(f_label)
        )
        
        self.wait(3)

        # Reset lecture line color for consistency
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
