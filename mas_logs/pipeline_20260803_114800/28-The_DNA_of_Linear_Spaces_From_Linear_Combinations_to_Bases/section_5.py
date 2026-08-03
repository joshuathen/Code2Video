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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The Basis: The Goldilocks Set"
        lines = [
            "A basis is a perfect set of building blocks.",
            "It must span the entire vector space.",
            "Every vector in a basis must be linearly independent.",
            "It is the minimal set needed for full coverage.",
            "A basis provides efficient navigation with no redundancy."
        ]
        self.setup_layout(title, lines)

        # Vector Definitions
        east_color = "#FFD700"
        north_color = "#00BFFF"
        up_color = "#ADFF2F"
        redundant_color = RED

        # Coordinate axes simulation in 2D (Perspective-like)
        origin = self.grid["D3"]
        v_east = Arrow(start=origin, end=origin + RIGHT*1.2, color=east_color, buff=0)
        v_north = Arrow(start=origin, end=origin + UP*1.2, color=north_color, buff=0)
        v_up = Arrow(start=origin, end=origin + (RIGHT*0.5 + UP*0.5), color=up_color, buff=0)
        
        label_e = Text("East", font_size=16, color=east_color)
        label_n = Text("North", font_size=16, color=north_color)
        label_u = Text("Up", font_size=16, color=up_color)
        
        self.place_at_grid(label_e, "D4", scale_factor=1.0)
        self.place_at_grid(label_n, "C3", scale_factor=1.0)
        # Issue 25: Move label_u from C4 to C5, scale 0.8
        self.place_at_grid(label_u, "C5", scale_factor=0.8)

        basis_group = VGroup(v_east, v_north, v_up, label_e, label_n, label_u)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(v_east), Create(v_north), Create(v_up), run_time=1.5)
        self.play(Write(label_e), Write(label_n), Write(label_u))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Representing "Spanning 3D volume" with a semi-transparent parallelepiped
        p1 = origin
        p2 = origin + RIGHT*1.2
        p3 = origin + RIGHT*1.2 + UP*1.2
        p4 = origin + UP*1.2
        p5 = origin + (RIGHT*0.5 + UP*0.5)
        p6 = p2 + (RIGHT*0.5 + UP*0.5)
        p7 = p3 + (RIGHT*0.5 + UP*0.5)
        p8 = p4 + (RIGHT*0.5 + UP*0.5)

        # Faces of the box
        faces = VGroup(
            Polygon(p1, p2, p3, p4), # front
            Polygon(p1, p2, p6, p5), # bottom
            Polygon(p2, p3, p7, p6), # right
            Polygon(p3, p4, p8, p7), # top
            Polygon(p4, p1, p5, p8), # left
            Polygon(p5, p6, p7, p8)  # back
        ).set_style(fill_opacity=0.2, fill_color=WHITE, stroke_width=1, stroke_color=WHITE)
        
        volume_box = faces
        
        self.play(FadeIn(volume_box))
        self.play(volume_box.animate.scale(1.1).set_opacity(0.1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Redundant Northeast vector
        v_ne = Arrow(start=origin, end=origin + RIGHT*0.8 + UP*0.8, color=redundant_color, buff=0)
        label_ne = Text("Northeast", font_size=16, color=redundant_color)
        # Issue 24: Move label_ne from D2 to D3, scale 0.8
        self.place_at_grid(label_ne, "D3", scale_factor=0.8)
        
        self.play(Create(v_ne), Write(label_ne))
        self.play(Indicate(v_ne, color=RED))
        self.play(FadeOut(v_ne), FadeOut(label_ne)) 
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Minimal set: Remove 'Up' and show volume collapse
        self.play(FadeOut(v_up), FadeOut(label_u))
        collapse_msg = Text("Volume -> Plane", font_size=20, color=RED)
        # Issue 26: Move collapse_msg from B4 to B5, scale 0.8
        self.place_at_grid(collapse_msg, "B5", scale_factor=0.8)
        
        # Plane representing the collapse
        collapsed_plane = Polygon(p1, p2, p3, p4, fill_opacity=0.5, fill_color=RED_E, stroke_width=2)
        
        self.play(FadeOut(volume_box), FadeIn(collapsed_plane), Write(collapse_msg))
        self.wait(1)
        
        # Restore Basis
        self.play(FadeIn(v_up), FadeIn(label_u), FadeOut(collapse_msg), FadeIn(volume_box), FadeOut(collapsed_plane))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(Indicate(basis_group))
        self.play(FadeOut(volume_box))
        self.wait(2)
