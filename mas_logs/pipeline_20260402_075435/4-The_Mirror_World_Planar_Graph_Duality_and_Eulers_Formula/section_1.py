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
        # Initialize the scene with title and the updated 5 lecture lines
        self.setup_layout(
            "Prerequisites: The Anatomy of a Planar Map", 
            [
                "A planar graph has edges that never cross.",
                "Green dots represent the vertices of our map.",
                "Cyan paths connect these vertices, forming edges.",
                "Magenta regions inside the graph are bounded faces.",
                "The vast outside area also counts as a face."
            ]
        )
        
        # Define Graph Geometry
        # We create a simple square graph (4 vertices, 4 edges)
        v1 = Dot(radius=0.1, color=WHITE)
        v2 = Dot(radius=0.1, color=WHITE)
        v3 = Dot(radius=0.1, color=WHITE)
        v4 = Dot(radius=0.1, color=WHITE)
        
        # Apply scaling and positioning via grid system (Issue 32)
        self.place_at_grid(v1, "B2", scale_factor=0.8)
        self.place_at_grid(v2, "B5", scale_factor=0.8)
        self.place_at_grid(v3, "E5", scale_factor=0.8)
        self.place_at_grid(v4, "E2", scale_factor=0.8)
        
        vertices = VGroup(v1, v2, v3, v4)
        
        e1 = Line(v1.get_center(), v2.get_center(), color=WHITE)
        e2 = Line(v2.get_center(), v3.get_center(), color=WHITE)
        e3 = Line(v3.get_center(), v4.get_center(), color=WHITE)
        e4 = Line(v4.get_center(), v1.get_center(), color=WHITE)
        
        edges = VGroup(e1, e2, e3, e4)
        
        # === Animation for Lecture Line 1 ===
        # Fade in a simple planar graph with 4 vertices and 4 edges.
        self.play(FadeIn(vertices), Create(edges), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the vertices in green (#00FF00) and add 'V' labels.
        v_labels = VGroup(
            Text("V", font_size=20, color="#00FF00").next_to(v1, UL, buff=0.1),
            Text("V", font_size=20, color="#00FF00").next_to(v2, UR, buff=0.1),
            Text("V", font_size=20, color="#00FF00").next_to(v3, DR, buff=0.1),
            Text("V", font_size=20, color="#00FF00").next_to(v4, DL, buff=0.1)
        )
        
        self.play(
            self.lecture[1].animate.set_color("#00FF00"),
            vertices.animate.set_color("#00FF00"),
            FadeIn(v_labels),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pulse the edges in cyan (#00FFFF) and add 'E' labels.
        e_labels = VGroup(
            Text("E", font_size=20, color="#00FFFF").next_to(e1, UP, buff=0.1),
            Text("E", font_size=20, color="#00FFFF").next_to(e2, RIGHT, buff=0.1),
            Text("E", font_size=20, color="#00FFFF").next_to(e3, DOWN, buff=0.1),
            Text("E", font_size=20, color="#00FFFF").next_to(e4, LEFT, buff=0.1)
        )
        
        self.play(
            self.lecture[2].animate.set_color("#00FFFF"),
            edges.animate.set_color("#00FFFF"),
            FadeIn(e_labels),
            run_time=1
        )
        # Pulse effect
        self.play(
            edges.animate.set_stroke(width=8),
            rate_func=there_and_back,
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Shade the bounded regions in semi-transparent magenta (#FF00FF).
        face_fill = Polygon(
            v1.get_center(), v2.get_center(), v3.get_center(), v4.get_center(),
            fill_color="#FF00FF", fill_opacity=0.4, stroke_width=0
        )
        
        self.play(
            self.lecture[3].animate.set_color("#FF00FF"),
            FadeIn(face_fill),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Label the background area as 'Infinite Face (F)'.
        # Fix: Horizontal centering and grid best practices (Issue 31)
        inf_face_label = Text("Infinite Face (F)", font_size=24, color="#FF00FF")
        self.place_in_area(inf_face_label, "F2", "F5", scale_factor=0.8)
        
        self.play(
            self.lecture[4].animate.set_color("#FF00FF"),
            Write(inf_face_label),
            run_time=1
        )
        self.wait(2)
