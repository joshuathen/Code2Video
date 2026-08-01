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
        # Initial Setup
        title_text = "Prerequisite: The Anatomy of a Planar Graph"
        lecture_lines = [
            "Planar graphs are drawn without any edges crossing.",
            "They consist of vertices, edges, and faces.",
            "The surrounding exterior area also counts as a face.",
            "This house graph has five vertices and six edges.",
            "Including the exterior, it contains exactly three faces."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Planar graphs are drawn without any edges crossing.
        v_a = Dot(self.grid["B3"])
        v_b = Dot(self.grid["B5"])
        v_c = Dot(self.grid["D3"])
        v_d = Dot(self.grid["D5"])
        
        e1 = Line(self.grid["B3"], self.grid["D5"], color=WHITE)
        e2 = Line(self.grid["B5"], self.grid["D3"], color=WHITE)
        
        crossing_group = VGroup(v_a, v_b, v_c, v_d, e1, e2)
        self.play(Create(crossing_group))
        self.wait(1)
        
        # Morph to non-crossing
        self.play(
            v_b.animate.move_to(self.grid["B6"]),
            v_d.animate.move_to(self.grid["D6"]),
            e1.animate.set_points_as_corners([self.grid["B3"], self.grid["D6"]]),
            e2.animate.set_points_as_corners([self.grid["B6"], self.grid["D3"]]),
            run_time=1.5
        )
        self.wait(1)
        self.play(FadeOut(crossing_group))

        # === Animation for Lecture Line 2 ===
        # They consist of vertices, edges, and faces.
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        
        v1_pos = self.grid["D2"]
        v2_pos = self.grid["D5"]
        v3_pos = self.grid["B2"]
        v4_pos = self.grid["B5"]
        v5_pos = (self.grid["A3"] + self.grid["A4"]) / 2
        
        v_points = [v1_pos, v2_pos, v4_pos, v3_pos, v5_pos]
        vertices = VGroup(*[Dot(p, color="#0000FF", radius=0.12) for p in v_points])
        
        edge_indices = [(0,1), (1,2), (2,3), (3,0), (3,4), (2,4)]
        edges = VGroup(*[Line(v_points[i], v_points[j], color=WHITE, stroke_width=4) for i, j in edge_indices])
        
        self.play(Create(vertices), Create(edges))
        v_label = Text("V = 5", font_size=24, color="#0000FF")
        self.place_at_grid(v_label, "E2")
        self.play(Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The surrounding exterior area also counts as a face.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        exterior_rect = SurroundingRectangle(VGroup(vertices, edges), buff=0.8, color="#00FF00", fill_opacity=0.1)
        ext_text = Text("Exterior Face", font_size=20, color="#00FF00")
        # Fixed: Move ext_text to E6 to avoid overlap with summary at F1-F6
        self.place_at_grid(ext_text, "E6", scale_factor=0.8)
        
        self.play(Create(exterior_rect), Write(ext_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This house graph has five vertices and six edges.
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        self.play(edges.animate.set_color("#FFFF00").set_stroke(width=8))
        e_label = Text("E = 6", font_size=24, color="#FFFF00")
        # Fixed: Move e_label to E4 for better alignment and symmetry
        self.place_at_grid(e_label, "E4", scale_factor=0.8)
        self.play(Write(e_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Including the exterior, it contains exactly three faces.
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        face1 = Polygon(v1_pos, v2_pos, v4_pos, v3_pos, fill_color="#00FF00", fill_opacity=0.3, stroke_width=0)
        face2 = Polygon(v3_pos, v4_pos, v5_pos, fill_color="#00FF00", fill_opacity=0.5, stroke_width=0)
        
        f_label = Text("F = 3", font_size=24, color="#00FF00")
        self.place_at_grid(f_label, "C3")
        
        self.play(FadeIn(face1), FadeIn(face2), Write(f_label))
        
        # Display summary
        summary = Text("Planar: Non-crossing edges, 3 Components", font_size=20, color=WHITE)
        self.place_in_area(summary, "F1", "F6")
        self.play(Write(summary))
        self.wait(2)
