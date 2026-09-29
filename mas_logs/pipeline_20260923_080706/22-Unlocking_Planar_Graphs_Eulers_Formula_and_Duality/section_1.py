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
        self.setup_layout("Defining the Planar Graph", [
            "Planar graphs avoid any edge crossings.",
            "Vertices are dots; edges are lines.",
            "Faces are enclosed or infinite regions."
        ])
        
        # Load assets
        dots_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg")
        lines_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lines.svg")
        
        # Create Graph Elements
        v1 = Dot(color=BLUE)
        v2 = Dot(color=BLUE)
        v3 = Dot(color=BLUE)
        v4 = Dot(color=BLUE)
        
        graph_nodes = VGroup(v1, v2, v3, v4)
        # Position vertices using the grid
        self.place_at_grid(v1, 'B2', 1.0)
        self.place_at_grid(v2, 'B5', 1.0)
        self.place_at_grid(v3, 'E5', 1.0)
        self.place_at_grid(v4, 'E2', 1.0)
        
        edges = VGroup(
            Line(v1.get_center(), v2.get_center(), color=WHITE),
            Line(v2.get_center(), v3.get_center(), color=WHITE),
            Line(v3.get_center(), v4.get_center(), color=WHITE),
            Line(v4.get_center(), v1.get_center(), color=WHITE),
            Line(v1.get_center(), v3.get_center(), color=WHITE)
        )
        
        graph_object = VGroup(graph_nodes, edges)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Using asset elements instead of primitives where possible
        self.play(FadeIn(dots_asset), FadeIn(graph_nodes))
        self.play(FadeIn(lines_asset), Create(edges))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Apply layout improvement from issue 20/35
        self.place_in_area(graph_object, 'B3', 'E5', scale_factor=0.9)
        self.play(Indicate(graph_nodes), edges.animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        face1 = Polygon(v1.get_center(), v2.get_center(), v3.get_center(), color=GREEN, fill_opacity=0.3)
        
        # Apply layout improvement from issue 21/36
        self.place_at_grid(face1, 'D4', scale_factor=0.75)
        
        self.play(FadeIn(face1))
        self.wait(2)
