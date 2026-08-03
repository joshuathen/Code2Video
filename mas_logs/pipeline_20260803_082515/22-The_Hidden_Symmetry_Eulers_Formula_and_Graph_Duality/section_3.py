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
        title_text = "The Mirror World: Introducing the Dual Graph"
        lecture_lines = [
            "Every planar graph has a corresponding dual graph.",
            "Duality transforms faces into vertices and vertices into faces.",
            "This creates a shadow world with reversed geometric roles."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color(YELLOW)
        
        # Create a planar graph (square with diagonal) in White
        v1 = Dot(color="#FFFFFF")
        v2 = Dot(color="#FFFFFF")
        v3 = Dot(color="#FFFFFF")
        v4 = Dot(color="#FFFFFF")
        
        # Position according to Issue 26
        self.place_at_grid(v1, 'B2')
        self.place_at_grid(v2, 'B5')
        self.place_at_grid(v3, 'E5')
        self.place_at_grid(v4, 'E2')
        
        e1 = Line(v1.get_center(), v2.get_center(), color="#FFFFFF")
        e2 = Line(v2.get_center(), v3.get_center(), color="#FFFFFF")
        e3 = Line(v3.get_center(), v4.get_center(), color="#FFFFFF")
        e4 = Line(v4.get_center(), v1.get_center(), color="#FFFFFF")
        e5 = Line(v1.get_center(), v3.get_center(), color="#FFFFFF")
        
        original_graph = VGroup(v1, v2, v3, v4, e1, e2, e3, e4, e5)
        self.play(Create(original_graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight current line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Place red dots in centers of faces (2 inner, 1 outer)
        # Position according to Issue 26 and 27
        dv1 = Dot(color="#FF0000") # Face 1 (upper triangle)
        dv2 = Dot(color="#FF0000") # Face 2 (lower triangle)
        dv3 = Dot(color="#FF0000") # Face 3 (outer face)
        
        self.place_at_grid(dv1, 'C4')
        self.place_at_grid(dv2, 'D3')
        self.place_at_grid(dv3, 'A4')
        
        dv_label = MathTex("V^*", color="#FF0000", font_size=24)
        self.place_at_grid(dv_label, 'A5', scale_factor=0.8)
        
        # Create "glowing" effect using Flash or just a simple fade-in
        self.play(
            FadeIn(dv1), FadeIn(dv2), FadeIn(dv3),
            Write(dv_label)
        )
        self.play(Flash(dv1, color="#FF0000"), Flash(dv2, color="#FF0000"), Flash(dv3, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight current line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fade original graph to grey
        self.play(
            original_graph.animate.set_color("#555555")
        )
        
        # Explanatory label positioned according to Issue 28
        face_rel_label = Text("1 Face = 1 Dual Vertex", font_size=20, color=YELLOW)
        self.place_in_area(face_rel_label, 'F3', 'F5', scale_factor=0.8)
        
        # Highlight relationship with a circle around one face/vertex
        # dv1 is at C4
        highlight_circle = Circle(radius=0.5, color=YELLOW).move_to(self.grid['C4'])
        
        self.play(
            Create(highlight_circle),
            Write(face_rel_label)
        )
        self.wait(3)
        
        # Finish
        self.lecture[2].set_color(WHITE)
        self.play(FadeOut(highlight_circle))
        self.wait(2)
