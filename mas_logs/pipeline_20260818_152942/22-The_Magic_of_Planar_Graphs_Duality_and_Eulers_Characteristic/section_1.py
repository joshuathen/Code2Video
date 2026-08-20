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
        self.setup_layout("Prerequisites: What is a Planar Graph?", 
                          ["Planar graphs are drawn without edge crossings.", 
                           "Key components are Vertices, Edges, and Faces.", 
                           "Untangle crossings to reveal planar embeddings."])
        
        # Asset Placeholder
        # Since the provided icon path is just a placeholder, we use a simple Shape
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Circle(radius=0.3, color=BLUE)

        # Create Graph
        nodes = VGroup(*[Dot(color="#FF6600") for _ in range(4)])
        # Use B4 for graph center as requested (Fix 20)
        self.place_at_grid(nodes[0], 'B3', scale_factor=0.6)
        self.place_at_grid(nodes[1], 'B5', scale_factor=0.6)
        self.place_at_grid(nodes[2], 'E3', scale_factor=0.6)
        self.place_at_grid(nodes[3], 'E5', scale_factor=0.6)
        
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color=WHITE),
            Line(nodes[1].get_center(), nodes[3].get_center(), color=WHITE),
            Line(nodes[3].get_center(), nodes[2].get_center(), color=WHITE),
            Line(nodes[2].get_center(), nodes[0].get_center(), color=WHITE),
            Line(nodes[0].get_center(), nodes[3].get_center(), color=WHITE) # Diagonal
        )
        graph_group = VGroup(nodes, edges)

        # === Animation for Lecture Line 1 ===
        # Show graph
        self.play(FadeIn(graph_group), FadeIn(asset_icon.move_to(self.grid['A6'])))
        self.lecture[0].set_color("#FF6600")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        crossing_edge = Line(nodes[1].get_center(), nodes[2].get_center(), color="#FF0000")
        self.play(Create(crossing_edge))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        planar_nodes = VGroup(*[Dot(color="#00FF00") for _ in range(4)])
        # Using area A3-F6 for grid layout (Fix 22)
        grid_points = VGroup(*[Dot(radius=0.05, color=GREY) for _ in range(36)])
        for i, pos in enumerate(self.grid.keys()):
            grid_points[i].move_to(self.grid[pos])
        self.place_in_area(grid_points, 'A3', 'F6', scale_factor=0.7)
        
        planar_edges = VGroup(
            Line(planar_nodes[0].get_center(), planar_nodes[1].get_center(), color=WHITE),
            Line(planar_nodes[1].get_center(), planar_nodes[3].get_center(), color=WHITE),
            Line(planar_nodes[3].get_center(), planar_nodes[2].get_center(), color=WHITE),
            Line(planar_nodes[2].get_center(), planar_nodes[0].get_center(), color=WHITE),
            Line(planar_nodes[1].get_center(), nodes[0].get_center() + np.array([0.5, 0.5, 0]), color=WHITE)
        )
        
        self.play(
            FadeOut(graph_group), FadeOut(crossing_edge),
            FadeIn(planar_nodes), FadeIn(planar_edges), FadeOut(asset_icon)
        )
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
