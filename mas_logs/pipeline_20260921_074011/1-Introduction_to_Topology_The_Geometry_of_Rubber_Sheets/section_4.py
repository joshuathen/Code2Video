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
        self.setup_layout("Real-world Applications", ["Topology applies to real-world science.", "It helps understand protein folding.", "Knots affect molecular function."])
        
        # Load Assets
        protein_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/protein.svg"
        
        # Animation Elements
        nodes = VGroup(*[SVGMobject(protein_path, color=BLUE).scale(0.5) for _ in range(6)])
        
        # Position nodes manually based on grid or create a network
        nodes[0].move_to(self.grid['B3'])
        nodes[1].move_to(self.grid['B5'])
        nodes[2].move_to(self.grid['D3'])
        nodes[3].move_to(self.grid['D5'])
        nodes[4].move_to(self.grid['F3'])
        nodes[5].move_to(self.grid['F5'])
        
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center()),
            Line(nodes[1].get_center(), nodes[2].get_center()),
            Line(nodes[2].get_center(), nodes[3].get_center()),
            Line(nodes[3].get_center(), nodes[4].get_center()),
            Line(nodes[4].get_center(), nodes[5].get_center()),
            Line(nodes[5].get_center(), nodes[0].get_center()),
        ).set_stroke(width=2, color=WHITE)
        
        network = VGroup(nodes, edges)
        # Apply the final requested fix from VideoCritic
        self.place_in_area(network, 'A3', 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#9B59B6"))
        self.play(Create(network), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E67E22"))
        path = VGroup(edges[0], edges[2], edges[4])
        self.play(path.animate.set_color(YELLOW), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        # Highlight node
        self.play(nodes[1].animate.scale(1.5), run_time=1)
