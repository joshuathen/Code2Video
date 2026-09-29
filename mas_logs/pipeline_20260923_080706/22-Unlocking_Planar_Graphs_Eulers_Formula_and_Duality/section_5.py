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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Duality aids in network resilience analysis.",
            "Identify redundant paths in infrastructure maps.",
            "Reinforce critical junctions to improve reliability."
        ]
        self.setup_layout("Application: Network Resilience", lecture_lines)
        
        # Load Assets
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        router_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/router.svg")
        cable_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cables.svg")
        
        # Define Network Components
        s1 = server_icon.copy().scale(0.3)
        s2 = server_icon.copy().scale(0.3)
        r1 = router_icon.copy().scale(0.3)
        r2 = router_icon.copy().scale(0.3)
        
        # Setup Diagram Nodes
        infrastructure_diagram = VGroup(s1, s2, r1, r2)
        self.place_at_grid(s1, 'B4', scale_factor=0.5)
        self.place_at_grid(s2, 'D4', scale_factor=0.5)
        self.place_at_grid(r1, 'B6', scale_factor=0.5)
        self.place_at_grid(r2, 'D6', scale_factor=0.5)
        
        # Create edges
        edge1 = Line(s1.get_center(), r1.get_center(), color=WHITE)
        edge2 = Line(r1.get_center(), r2.get_center(), color=WHITE)
        edge3 = Line(r2.get_center(), s2.get_center(), color=WHITE)
        edge4 = Line(s2.get_center(), s1.get_center(), color=WHITE)
        redundant_path_line = Line(s1.get_center(), r2.get_center(), color=BLUE)
        
        graph = VGroup(infrastructure_diagram, edge1, edge2, edge3, edge4)
        
        # Fix for overlap: self.place_in_area(infrastructure_diagram, 'A4', 'F6', scale_factor=0.9)
        self.place_in_area(graph, 'A4', 'F6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(graph))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix for diagonal line clearance: self.place_in_area(redundant_path_line, 'B4', 'E5', scale_factor=0.8)
        self.place_in_area(redundant_path_line, 'B4', 'E5', scale_factor=0.8)
        self.play(Create(redundant_path_line))
        self.play(FadeOut(redundant_path_line))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        critical_node = Dot(color=GREEN)
        # Fix for critical node positioning: self.place_at_grid(critical_node, 'C5', scale_factor=0.7)
        self.place_at_grid(critical_node, 'C5', scale_factor=0.7)
        self.play(FadeIn(critical_node))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
