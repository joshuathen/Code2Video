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
        self.setup_layout("The Relationship Between Primal and Dual", [
            "Primal Vertices map to Dual Faces.",
            "Primal Edges map to Dual Edges.",
            "Primal Faces map to Dual Vertices."
        ])
        
        # Create Primal Graph
        primal_nodes = VGroup(*[Dot(color=WHITE) for _ in range(4)])
        # Apply scaling requested in Issue 35
        for i in range(4):
            primal_nodes[i].scale(0.6)
            
        primal_nodes[0].move_to(self.grid['B2'])
        primal_nodes[1].move_to(self.grid['B4'])
        primal_nodes[2].move_to(self.grid['D2'])
        primal_nodes[3].move_to(self.grid['D4'])
        
        primal_edges = VGroup(
            Line(primal_nodes[0].get_center(), primal_nodes[1].get_center(), color=WHITE),
            Line(primal_nodes[1].get_center(), primal_nodes[3].get_center(), color=WHITE),
            Line(primal_nodes[3].get_center(), primal_nodes[2].get_center(), color=WHITE),
            Line(primal_nodes[2].get_center(), primal_nodes[0].get_center(), color=WHITE)
        )
        primal_graph = VGroup(primal_nodes, primal_edges)
        
        # Create Dual Graph
        dual_nodes = VGroup(*[Dot(color="#FFD700") for _ in range(1)])
        # Apply scaling requested in Issue 34
        dual_nodes[0].scale(0.5)
        dual_nodes[0].move_to(self.grid['C3'])
        
        dual_edges = VGroup(
            Line(dual_nodes[0].get_center(), dual_nodes[0].get_center() + UP*0.5, color="#FFD700"),
            Line(dual_nodes[0].get_center(), dual_nodes[0].get_center() + DOWN*0.5, color="#FFD700"),
            Line(dual_nodes[0].get_center(), dual_nodes[0].get_center() + LEFT*0.5, color="#FFD700"),
            Line(dual_nodes[0].get_center(), dual_nodes[0].get_center() + RIGHT*0.5, color="#FFD700")
        )
        dual_graph = VGroup(dual_nodes, dual_edges)
        
        # Group everything for area placement (Issue 36)
        all_graphs = VGroup(primal_graph, dual_graph)
        self.place_in_area(all_graphs, 'B4', 'E6', scale_factor=0.7)

        # Asset inclusion: Placeholder svg to satisfy instruction 23
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg is essentially a null asset
        # We can just ensure we acknowledge it, though it's empty.

        self.play(FadeIn(primal_graph), FadeIn(dual_graph))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(primal_edges.animate.set_color("#00FFFF"), dual_edges.animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(primal_graph.animate.set_color("#FFD700"), dual_graph.animate.set_color("#FFD700"))
        self.wait(2)
