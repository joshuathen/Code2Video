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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Interplay of Duality and Euler", ["Dual vertices map to original faces.", "Edges remain the same in number.", "Faces become vertices in the dual."])
        
        # Original G (triangle)
        v1 = Dot(self.grid["B3"], color=BLUE)
        v2 = Dot(self.grid["D2"], color=BLUE)
        v3 = Dot(self.grid["D4"], color=BLUE)
        e1 = Line(v1.get_center(), v2.get_center(), color=WHITE)
        e2 = Line(v2.get_center(), v3.get_center(), color=WHITE)
        e3 = Line(v3.get_center(), v1.get_center(), color=WHITE)
        triangle_group = VGroup(e1, e2, e3, v1, v2, v3)
        self.place_in_area(triangle_group, 'B4', 'E6', scale_factor=0.8)
        
        # Assets
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        poly_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/polyhedron.svg")
        
        # Dual vertex
        dual_v = Dot(self.grid["C3"], color=YELLOW)
        yellow_dot_label = Text("Dual Vertex", font_size=16, color=YELLOW)
        self.place_at_grid(yellow_dot_label, 'D5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(graph_icon, 'B3', scale_factor=0.5)
        self.play(FadeIn(graph_icon), FadeIn(dual_v), Write(yellow_dot_label))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        edge_label = Text("E = E*", font_size=20, color=WHITE)
        self.place_at_grid(edge_label, 'C4', scale_factor=0.9)
        self.play(Write(edge_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        face_label = Text("F = V*", font_size=20, color=WHITE)
        self.place_at_grid(face_label, 'D4', scale_factor=0.9)
        self.place_at_grid(poly_icon, 'E3', scale_factor=0.5)
        self.play(Write(face_label), FadeIn(poly_icon))
        self.wait(2)
