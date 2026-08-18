from manim import *

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
        self.setup_layout("Prerequisites & Definitions", [
            "Planar graphs avoid edge crossings.",
            "Vertices, edges, and faces are fundamental components.",
            "Visualize a simple triangular planar graph."
        ])
        
        # Setup the graph elements
        v1 = Dot(color=WHITE)
        v2 = Dot(color=WHITE)
        v3 = Dot(color=WHITE)
        
        # Positions
        v1.move_to(self.grid["B3"])
        v2.move_to(self.grid["D2"])
        v3.move_to(self.grid["D4"])
        
        e1 = Line(v1.get_center(), v2.get_center(), color="#00FFFF")
        e2 = Line(v2.get_center(), v3.get_center(), color="#00FFFF")
        e3 = Line(v3.get_center(), v1.get_center(), color="#00FFFF")
        
        graph_elements = VGroup(v1, v2, v3, e1, e2, e3)
        face_inner = Polygon(v1.get_center(), v2.get_center(), v3.get_center(), color="#FF00FF", fill_opacity=0.3)
        
        graph_all = VGroup(graph_elements, face_inner)
        self.place_in_area(graph_all, 'B3', 'E5', scale_factor=0.9)
        
        label_v = Text("V", font_size=20, color=WHITE)
        label_e = Text("E", font_size=20, color="#00FFFF")
        label_f = Text("F", font_size=20, color="#FF00FF")
        
        self.place_at_grid(label_v, 'B4', scale_factor=0.7)
        self.place_at_grid(label_e, 'D2', scale_factor=0.7)
        self.place_at_grid(label_f, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(v1), Create(v2), Create(v3), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(e1), Create(e2), Create(e3), Write(label_e))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(face_inner), Write(label_f))
        self.wait(2)
