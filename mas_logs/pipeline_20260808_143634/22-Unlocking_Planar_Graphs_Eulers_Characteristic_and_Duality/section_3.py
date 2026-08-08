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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Concept of Dual Graphs", [
            "Dual graph transformation: Place vertices in faces.", 
            "Connect vertices if faces share an edge.", 
            "Analogy: Sensors in rooms connect through doors."
        ])
        
        # Assets
        room_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/room.svg"
        sensor_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg"
        door_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/door.svg"

        # Construct original graph using SVG as nodes
        # Simplified graph
        nodes_pos = {"v1": "B4", "v2": "B5", "v3": "D4", "v4": "D5"}
        node_mobs = {k: SVGMobject(room_asset, color=WHITE).scale(0.3) for k in nodes_pos}
        for k, pos in nodes_pos.items():
            node_mobs[k].move_to(self.grid[pos])
            
        edges = [("v1", "v2"), ("v2", "v4"), ("v4", "v3"), ("v3", "v1"), ("v1", "v4")]
        edge_mobs = VGroup(*[Line(node_mobs[u].get_center(), node_mobs[v].get_center(), color=WHITE) for u, v in edges])
        
        graph_group = VGroup(*node_mobs.values(), edge_mobs)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(graph_group))
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        
        dual_nodes = [
            SVGMobject(sensor_asset, color="#FFA500").scale(0.3).move_to(self.grid["C4"]),
            SVGMobject(sensor_asset, color="#FFA500").scale(0.3).move_to(self.grid["C5"])
        ]
        dual_group = VGroup(*dual_nodes)
        self.play(FadeIn(dual_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        dual_edge = SVGMobject(door_asset, color="#FFFF00").scale(0.2).move_to((dual_nodes[0].get_center() + dual_nodes[1].get_center()) / 2)
        self.play(FadeIn(dual_edge))
        
        dual_final_group = VGroup(dual_group, dual_edge)
        self.place_in_area(dual_final_group, 'B2', 'E5', scale_factor=1.0)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
