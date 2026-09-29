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
        self.setup_layout("Real-World Topology", ["Topology categorizes complex shapes.", "Used in DNA and robotics.", "Persistence defines the object."])
        
        # Load SVG asset
        city_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/city.svg")
        
        # 1. Map (Represented by city icon + nodes)
        self.place_in_area(city_icon, 'A4', 'F6', scale_factor=0.9)
        
        # Grid visual points (nodes for network)
        city_points = VGroup(*[Dot(self.grid[pos], radius=0.08, color=WHITE) for pos in ["B3", "B5", "D3", "D5", "F4"]])
        city_lines = VGroup(
            Line(city_points[0].get_center(), city_points[1].get_center(), color=GRAY),
            Line(city_points[0].get_center(), city_points[2].get_center(), color=GRAY),
            Line(city_points[1].get_center(), city_points[3].get_center(), color=GRAY),
            Line(city_points[2].get_center(), city_points[4].get_center(), color=GRAY),
            Line(city_points[3].get_center(), city_points[4].get_center(), color=GRAY)
        )
        map_network = VGroup(city_icon, city_points, city_lines)
        
        # Label for a node
        node_label = Text("Node", font_size=16, color=YELLOW)
        self.place_at_grid(node_label, 'A3', scale_factor=0.6)
        
        # Topological Graph
        graph_network = VGroup(city_points.copy(), city_lines.copy()).set_color("#00CED1")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(map_network), FadeIn(node_label))
        self.lecture[0].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(TransformFromCopy(map_network, graph_network))
        self.lecture[1].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        data_pulse = Circle(radius=0.15, color=GOLD).move_to(city_points[0].get_center())
        self.play(Create(data_pulse))
        self.play(MoveAlongPath(data_pulse, city_lines[0]), run_time=1.5)
        self.play(FadeOut(data_pulse))
        self.lecture[2].set_color("#00CED1")
        self.wait(2)
