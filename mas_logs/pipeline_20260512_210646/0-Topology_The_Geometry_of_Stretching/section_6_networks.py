from manim import *
import numpy as np

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

class Section6NetworksScene(TeachingScene):
    def construct(self):
        title_str = "Topology in Connectivity: Graphs"
        lines_str = [
            "Topology also applies to connections in a network.",
            "Map distances matter less than how nodes are linked.",
            "The London Underground map illustrates this topological simplicity."
        ]
        self.setup_layout(title_str, lines_str)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#708090"))
        
        # Define 6 initial nodes (dots)
        # We'll place them roughly and then apply layout constraints
        dots = VGroup(*[Dot(color=WHITE) for _ in range(6)])
        
        # Create initial curved edges
        # We'll use get_center() later to define arcs correctly after positioning
        curves = VGroup()
        
        # Group everything as network_graph for positioning
        network_graph = VGroup(dots, curves)
        
        # [Issue 48] Fix: Positioning initial network graph
        self.place_in_area(network_graph, 'A2', 'F6', scale_factor=0.8)
        
        # [Issue 49] Fix: Anchor central hub node to C4
        central_node = dots[0] # designated central node
        self.place_at_grid(central_node, 'C4', scale_factor=1.0)
        
        # Distribute other nodes manually based on grid for starting positions
        dots[1].move_to(self.grid['B2'])
        dots[2].move_to(self.grid['B5'])
        dots[3].move_to(self.grid['D1'])
        dots[4].move_to(self.grid['E6'])
        dots[5].move_to(self.grid['F2'])
        
        # Now define the arcs connecting them
        p_hub = central_node.get_center()
        p1, p2, p3, p4, p5 = [dots[i].get_center() for i in range(1, 6)]
        
        c_list = [
            ArcBetweenPoints(p1, p2, angle=TAU/4, color="#708090"),
            ArcBetweenPoints(p1, p_hub, angle=-TAU/6, color="#708090"),
            ArcBetweenPoints(p3, p_hub, angle=TAU/8, color="#708090"),
            ArcBetweenPoints(p_hub, p4, angle=-TAU/12, color="#708090"),
            ArcBetweenPoints(p_hub, p5, angle=TAU/6, color="#708090"),
            ArcBetweenPoints(p2, p4, angle=TAU/5, color="#708090")
        ]
        curves.add(*c_list)

        self.play(Create(dots, run_time=1), Create(curves, run_time=1.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#1E90FF")
        )

        # [Issue 35] Asset Integration
        subway_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/subway.svg")
        self.place_at_grid(subway_icon, 'A6', scale_factor=0.4)

        # Straight lines in subway colors (blue, red, green)
        # Blue line: p1-p_hub-p2
        s1 = Line(p1, p_hub, color="#1E90FF", stroke_width=6)
        s2 = Line(p_hub, p2, color="#1E90FF", stroke_width=6)
        # Red line: p3-p_hub-p4
        s3 = Line(p3, p_hub, color="#FF4500", stroke_width=6)
        s4 = Line(p_hub, p4, color="#FF4500", stroke_width=6)
        # Green line: p5-p_hub
        s5 = Line(p5, p_hub, color="#32CD32", stroke_width=6)
        s6 = Line(p1, p2, color="#32CD32", stroke_width=6) # auxiliary connection
        
        straights = VGroup(s1, s2, s3, s4, s5, s6)
        straightened_graph = VGroup(dots, straights)
        
        # [Issue 50] Fix: Resize/position straightened graph to avoid cutoff
        self.place_in_area(straightened_graph, 'B2', 'E5', scale_factor=0.7)

        self.play(
            ReplacementTransform(curves, straights),
            FadeIn(subway_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )

        # Pulse nodes in yellow (#FFFF00) to show connectivity preserved
        self.play(
            LaggedStart(
                *[Indicate(d, color="#FFFF00", scale_factor=2.0) for d in dots],
                lag_ratio=0.15
            )
        )
        
        self.wait(2)
