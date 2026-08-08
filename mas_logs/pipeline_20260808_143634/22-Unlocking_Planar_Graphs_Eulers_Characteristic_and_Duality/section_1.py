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
        self.setup_layout("Prerequisite: Defining the Planar Graph", [
            "A planar graph has no crossing edges.",
            "Components: Vertices, Edges, and Faces.",
            "Example: Spider-web graph structure."
        ])

        # Define graph components
        v_coords = [self.grid['B3'], self.grid['B5'], self.grid['D2'], self.grid['D6'], self.grid['F3'], self.grid['F5']]
        vertices = VGroup(*[Dot(coord, color=WHITE) for coord in v_coords])
        edges = VGroup(
            Line(v_coords[0], v_coords[1], color=WHITE),
            Line(v_coords[1], v_coords[3], color=WHITE),
            Line(v_coords[3], v_coords[5], color=WHITE),
            Line(v_coords[5], v_coords[4], color=WHITE),
            Line(v_coords[4], v_coords[2], color=WHITE),
            Line(v_coords[2], v_coords[0], color=WHITE),
            Line(v_coords[0], v_coords[3], color=WHITE),
            Line(v_coords[2], v_coords[5], color=WHITE)
        )
        graph = VGroup(edges, vertices)
        
        # Asset: Spider icon
        spider_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spider.svg")
        self.place_in_area(spider_asset, 'A3', 'F6', scale_factor=0.5)
        
        graph_label = Text("G", font_size=24, color=WHITE).next_to(graph, UP)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), Create(graph), FadeIn(spider_asset), Write(graph_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        v_label = Text("V", font_size=20, color=RED)
        e_label = Text("E", font_size=20, color=GREEN)
        
        self.place_at_grid(v_label, 'D2', scale_factor=0.7)
        self.place_at_grid(e_label, 'D5', scale_factor=0.7)
        
        self.play(vertices.animate.set_color(RED), Write(v_label))
        self.play(edges.animate.set_color(GREEN), Write(e_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Highlight a face
        face = Polygon(v_coords[0], v_coords[1], v_coords[3], v_coords[2], color=BLUE, fill_opacity=0.3)
        f_label = Text("F", font_size=20, color=BLUE)
        self.place_at_grid(f_label, 'B4', scale_factor=0.7)
        
        self.play(FadeIn(face), Write(f_label))
        
        # Draw closed curve overlay using asset
        circle_overlay = Circle(color=PURPLE, stroke_width=4)
        self.place_in_area(circle_overlay, 'B3', 'E4', scale_factor=0.8)
        
        self.play(Create(circle_overlay))
        self.wait(2)
