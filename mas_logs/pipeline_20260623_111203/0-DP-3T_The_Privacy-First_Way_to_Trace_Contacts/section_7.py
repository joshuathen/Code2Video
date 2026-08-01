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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        self.setup_layout("Summary: Why it's Secure", [
            'The server never sees the actual contact graph.',
            "Matching is decentralized, occurring only on the user's phone.",
            'DP-3T ensures safety without compromising individual privacy.'
        ])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        
        # Contact Graph
        node_positions = ['B2', 'D2', 'B4', 'D4']
        nodes = VGroup(*[Circle(radius=0.15, color=BLUE_B, fill_opacity=1) for _ in range(4)])
        for node, pos in zip(nodes, node_positions):
            self.place_at_grid(node, pos)
            
        edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color=BLUE_B),
            Line(nodes[1].get_center(), nodes[3].get_center(), color=BLUE_B),
            Line(nodes[3].get_center(), nodes[2].get_center(), color=BLUE_B),
            Line(nodes[2].get_center(), nodes[0].get_center(), color=BLUE_B)
        )
        graph = VGroup(edges, nodes)
        
        # Server Icon
        server = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/server.svg")
        server.set_color("#95A5A6")
        self.place_at_grid(server, 'C1', scale_factor=0.6)
        
        # No Entry Sign
        no_entry_circle = Circle(radius=0.5, color="#E74C3C", stroke_width=8)
        no_entry_line = Line(start=0.35*LEFT + 0.35*UP, end=0.35*RIGHT + 0.35*DOWN, color="#E74C3C", stroke_width=8)
        no_entry = VGroup(no_entry_circle, no_entry_line)
        self.place_at_grid(no_entry, 'C3', scale_factor=1.2)
        
        self.play(Create(graph))
        self.play(FadeIn(server))
        self.play(FadeIn(no_entry))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW_B),
            FadeOut(graph), FadeOut(server), FadeOut(no_entry)
        )

        # Phone Icons
        phone1 = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg")
        phone2 = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg")
        phone1.set_color(WHITE)
        phone2.set_color(WHITE)
        
        self.place_at_grid(phone1, 'B2', scale_factor=0.8)
        self.place_at_grid(phone2, 'E4', scale_factor=0.8)

        # Gears
        def get_gear():
            gear_body = Circle(radius=0.2, color="#F1C40F", fill_opacity=1)
            teeth = VGroup(*[
                Rectangle(width=0.1, height=0.05, color="#F1C40F", fill_opacity=1)
                .move_to(0.22 * np.array([np.cos(a), np.sin(a), 0]))
                .rotate(a)
                for a in np.linspace(0, 2*PI, 8, endpoint=False)
            ])
            return VGroup(gear_body, teeth)

        gear1 = get_gear()
        gear2 = get_gear()
        
        # Place gears inside phones
        gear1.move_to(phone1.get_center())
        gear2.move_to(phone2.get_center())
        
        self.play(FadeIn(phone1), FadeIn(phone2))
        self.play(FadeIn(gear1), FadeIn(gear2))
        
        # Animate gear rotation
        gear1.add_updater(lambda m, dt: m.rotate(dt * 2))
        gear2.add_updater(lambda m, dt: m.rotate(-dt * 2))
        
        self.wait(2)
        gear1.clear_updaters()
        gear2.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN_B),
            FadeOut(phone1), FadeOut(phone2), FadeOut(gear1), FadeOut(gear2)
        )

        # Comparison Table
        table_title = Text("Privacy through Decentralization", font_size=20, color=WHITE)
        self.place_at_grid(table_title, 'A3', scale_factor=1.0)
        
        header_y = self.grid['B1'][1]
        row_y = self.grid['D1'][1]
        
        col1_x = self.grid['A2'][0]
        col2_x = self.grid['A3'][0] + 0.5
        col3_x = self.grid['A5'][0]
        
        headers = VGroup(
            Text("Centralized", font_size=18, color=WHITE).move_to([col2_x, header_y, 0]),
            Text("DP-3T", font_size=18, color=GREEN_B).move_to([col3_x, header_y, 0])
        )
        
        row_label = Text("Privacy", font_size=18, color=WHITE).move_to([col1_x, row_y, 0])
        
        # Symbols (Using manual construction to avoid LaTeX dependency)
        dummy_square = Square(side_length=0.4, stroke_opacity=0).move_to([col2_x, row_y, 0])
        red_x = Cross(dummy_square, stroke_width=6, stroke_color=RED).scale(0.5)
        
        check = VGroup(
            Line(LEFT * 0.15 + DOWN * 0.1, ORIGIN, stroke_width=6),
            Line(ORIGIN, RIGHT * 0.25 + UP * 0.35, stroke_width=6)
        ).move_to([col3_x, row_y, 0]).set_color("#2ECC71")
        
        # Table Lines
        h_line = Line([col1_x-0.5, header_y-0.4, 0], [col3_x+0.5, header_y-0.4, 0], color=WHITE)
        v_line1 = Line([col1_x+0.5, header_y+0.3, 0], [col1_x+0.5, row_y-0.3, 0], color=WHITE)
        
        table = VGroup(table_title, headers, row_label, red_x, check, h_line, v_line1)
        
        self.play(FadeIn(table_title))
        self.play(Write(headers), Write(row_label))
        self.play(Create(h_line), Create(v_line1))
        self.play(Create(red_x), Create(check))
        
        self.wait(2)
