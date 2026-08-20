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
        self.setup_layout("Introduction: The Map-Maker's Challenge", [
            "A planar graph has no crossing edges.",
            "We can redraw graphs to remove crossings.",
            "This is vital for subway map design."
        ])
        
        # Load asset
        subway_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/subway.svg")
        
        # Define the graph components
        nodes = [Dot(color=WHITE) for _ in range(4)]
        edges = [Line(nodes[i].get_center(), nodes[(i+1)%4].get_center(), color=WHITE) for i in range(4)]
        
        group_nodes = VGroup(*nodes)
        
        # Apply positioning requested in issue 19, 20, 21, 34, 35, 36
        self.place_at_grid(nodes[0], 'B3', scale_factor=0.6)
        self.place_at_grid(nodes[1], 'B5', scale_factor=0.6)
        self.place_at_grid(nodes[2], 'D5', scale_factor=0.6)
        self.place_at_grid(nodes[3], 'D3', scale_factor=0.6)
        
        # Recalculate edges after moving nodes
        new_edges = VGroup(*[Line(nodes[i].get_center(), nodes[(i+1)%4].get_center(), color=WHITE) for i in range(4)])
        graph = VGroup(new_edges, group_nodes)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), run_time=1)
        self.place_at_grid(subway_icon, 'C4', scale_factor=1.5)
        self.play(FadeIn(subway_icon), Create(graph), run_time=2)
        self.play(self.lecture[0].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), run_time=1)
        # Highlight vertices
        for node in nodes:
            self.play(node.animate.set_color("#00FF00"), run_time=0.3)
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]), run_time=1)
        # Highlight edges
        for edge in new_edges:
            self.play(edge.animate.set_color("#0000FF"), run_time=0.3)
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        
        # Fill face
        fill_shape = Polygon(*[n.get_center() for n in nodes], color="#CCCCCC", fill_opacity=0.5)
        self.play(FadeIn(fill_shape), run_time=1)
        
        self.wait(2)
