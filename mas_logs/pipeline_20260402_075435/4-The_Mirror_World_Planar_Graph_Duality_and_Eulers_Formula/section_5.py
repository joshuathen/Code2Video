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
        title = "Application: The Dual Perspective in Design"
        lines = [
            "Duality simplifies complex network and circuit designs.",
            "Solving dual problems provides original solutions.",
            "Mirror graphs reveal hidden paths in our physical world."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(BLUE_C))
        
        # Blueprint Asset
        blueprint_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/blueprint.svg")
        blueprint_svg.set_color(BLUE_D)
        self.place_in_area(blueprint_svg, "B2", "E5", scale_factor=1.2)
        
        # Primal Graph (Blueprint metaphor nodes/edges)
        p1 = self.grid["B2"]
        p2 = self.grid["B5"]
        p3 = self.grid["E5"]
        p4 = self.grid["E2"]
        p5 = (p1 + p3) / 2 # Center
        
        p_color = BLUE_D
        primal_dots = VGroup(
            Dot(p1, color=p_color), Dot(p2, color=p_color),
            Dot(p3, color=p_color), Dot(p4, color=p_color),
            Dot(p5, color=p_color)
        )
        
        primal_edges = VGroup(
            Line(p1, p2, color=p_color), Line(p2, p3, color=p_color),
            Line(p3, p4, color=p_color), Line(p4, p1, color=p_color),
            Line(p1, p5, color=p_color), Line(p2, p5, color=p_color),
            Line(p3, p5, color=p_color), Line(p4, p5, color=p_color)
        )
        
        primal_graph = VGroup(primal_edges, primal_dots)
        blueprint_label = Text("Blueprint (Primal)", font_size=20, color=BLUE_C)
        # Fix for Issue 42: Adjust label area
        self.place_in_area(blueprint_label, 'A1', 'A3', scale_factor=0.8)
        
        self.play(
            FadeIn(blueprint_svg),
            Create(primal_graph),
            FadeIn(blueprint_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Dim line 1, highlight line 2
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(ORANGE)
        )
        
        # Dual Graph Construction
        d_color = ORANGE
        # Face locations (approximate for the 4 triangles and the outer face)
        d1 = (p1 + p2 + p5) / 3 # Face 1
        d2 = (p2 + p3 + p5) / 3 # Face 2
        d3 = (p3 + p4 + p5) / 3 # Face 3
        d4 = (p4 + p1 + p5) / 3 # Face 4
        d_out = self.grid["C6"] + RIGHT * 0.4 # Outside Face
        
        dual_pts = [d1, d2, d3, d4, d_out]
        
        # Room icons at dual nodes
        room_icons = VGroup(*[
            SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/room.svg").set_color(ORANGE).scale(0.2).move_to(pt)
            for pt in dual_pts
        ])
        
        dual_edges = VGroup(
            Line(d1, d2, color=d_color), Line(d2, d3, color=d_color),
            Line(d3, d4, color=d_color), Line(d4, d1, color=d_color),
            Line(d1, d_out, color=d_color), Line(d2, d_out, color=d_color),
            Line(d3, d_out, color=d_color), Line(d4, d_out, color=d_color)
        )
        
        dual_label = Text("Dual Movement", font_size=20, color=ORANGE)
        # Fix for Issue 43: Adjust label area
        self.place_in_area(dual_label, 'A4', 'A6', scale_factor=0.8)
        
        # Dim the primal side and show dual movement
        self.play(
            blueprint_svg.animate.set_opacity(0.1),
            primal_graph.animate.set_opacity(0.1),
            Create(dual_edges),
            FadeIn(room_icons),
            FadeIn(dual_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Dim line 2, highlight line 3
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#00FF7F")
        )
        
        # Circuit board transformation
        pcb_trace_color = "#00FF7F" # Spring Green
        pcb_pad_color = "#E5E4E2"   # Platinum
        
        # Transform the dual graph into a circuit board layout
        pcb_edges = dual_edges.copy().set_color(pcb_trace_color).set_stroke(width=5)
        pcb_pads = VGroup(*[
            Circle(radius=0.1, color=pcb_pad_color, fill_opacity=1, fill_color=pcb_pad_color).move_to(pt)
            for pt in dual_pts
        ])
        
        circuit_label = Text("Circuit Design", font_size=20, color=pcb_trace_color)
        # Fix for Issue 44: Adjust label area
        self.place_in_area(circuit_label, 'A2', 'A5', scale_factor=0.8)

        self.play(
            FadeOut(blueprint_svg),
            FadeOut(primal_graph),
            FadeOut(blueprint_label),
            FadeOut(dual_label),
            ReplacementTransform(dual_edges, pcb_edges),
            ReplacementTransform(room_icons, pcb_pads),
            FadeIn(circuit_label)
        )
        
        # Final polish: highlight the paths
        self.play(pcb_edges.animate.set_stroke(width=8), run_time=1)
        self.play(pcb_edges.animate.set_stroke(width=5), run_time=1)
        
        self.wait(2)
