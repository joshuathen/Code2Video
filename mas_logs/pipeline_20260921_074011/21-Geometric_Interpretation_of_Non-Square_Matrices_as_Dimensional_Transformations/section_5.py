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
        self.setup_layout("Synthesis & Real-world Application", [
            "Matrices drive data science applications.", 
            "Expand data for better analysis.", 
            "Compress data for efficient storage."
        ])
        
        # Elements
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color="#FF69B4")
        harddrive_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/harddrive.svg", color="#FFA500")

        network_nodes = VGroup(*[Circle(radius=0.2, fill_opacity=0.8, color=WHITE) for _ in range(5)])
        
        for i, node in enumerate(network_nodes):
            self.place_at_grid(node, ["B2", "B4", "C1", "C3", "C5"][i])
        self.place_at_grid(server_icon, "D2", scale_factor=0.3)
            
        network_group = VGroup(network_nodes, server_icon)
        
        edges = VGroup(*[Line(network_nodes[i].get_center(), network_nodes[j].get_center(), stroke_width=2) 
                         for i, j in [(0, 1), (0, 2), (1, 3), (2, 4)]])
        
        # Use an animation container as requested by criticism 34
        animation_container = VGroup(network_group, edges)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(animation_container))
        self.play(self.lecture[0].animate.set_color("#FF69B4"))
        self.play(network_group.animate.set_color("#FF69B4"))

        # === Animation for Lecture Line 2 ===
        signal = Dot(color="#7FFF00")
        self.play(self.lecture[1].animate.set_color("#7FFF00"))
        self.add(signal)
        for i in range(len(edges)):
            self.play(MoveAlongPath(signal, edges[i]), run_time=0.5)
        self.remove(signal)

        # === Animation for Lecture Line 3 ===
        result_box = harddrive_icon
        self.place_at_grid(result_box, "D5", scale_factor=0.5)
        
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        self.play(Create(result_box), FadeOut(animation_container))
        self.wait(2)
