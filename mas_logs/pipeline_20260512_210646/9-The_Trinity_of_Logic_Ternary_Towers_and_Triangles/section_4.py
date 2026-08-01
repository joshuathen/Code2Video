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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup basic layout
        lines = [
            "Adding more disks expands the state-space graph recursively.",
            "Two disks form three connected triangles of states.",
            "Increasing disks reveals the iconic Sierpinski Triangle fractal.",
            "Corners represent all disks stacked on a single peg.",
            "The fractal structure perfectly captures the game's logic."
        ]
        self.setup_layout("Emergence of the Sierpinski Triangle", lines)

        # Helper to generate Level 2 graph (9 nodes)
        def create_graph_level_2(center, size):
            h = size * np.sqrt(3) / 2
            # 3 clusters of 3 nodes
            sub_size = size / 2
            
            # Centers for the 3 sub-triangles
            c_top = center + np.array([0, h/4, 0])
            c_bl = center + np.array([-size/4, -h/4, 0])
            c_br = center + np.array([size/4, -h/4, 0])
            
            def get_tri_verts(c, s):
                sh = s * np.sqrt(3) / 2
                return [c + np.array([0, sh/2, 0]), c + np.array([-s/2, -sh/2, 0]), c + np.array([s/2, -sh/2, 0])]

            v_top = get_tri_verts(c_top, sub_size/2)
            v_bl = get_tri_verts(c_bl, sub_size/2)
            v_br = get_tri_verts(c_br, sub_size/2)
            
            all_verts = v_top + v_bl + v_br
            nodes = VGroup(*[Dot(p, radius=0.06, color="#87CEEB") for p in all_verts])
            
            edges = VGroup()
            # Inner triangle edges
            for i in range(0, 9, 3):
                edges.add(Line(all_verts[i], all_verts[i+1], stroke_width=2, color=WHITE))
                edges.add(Line(all_verts[i+1], all_verts[i+2], stroke_width=2, color=WHITE))
                edges.add(Line(all_verts[i+2], all_verts[i], stroke_width=2, color=WHITE))
            
            # Bridge edges for Level 2
            edges.add(Line(all_verts[1], all_verts[3], stroke_width=2, color=WHITE))
            edges.add(Line(all_verts[2], all_verts[6], stroke_width=2, color=WHITE))
            edges.add(Line(all_verts[5], all_verts[7], stroke_width=2, color=WHITE))
            
            return VGroup(nodes, edges)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Show Level 2 graph (2-disk state space)
        g2 = create_graph_level_2(ORIGIN, 3.0)
        # Fix issue 33: Adjust position and scale
        self.place_in_area(g2, "B1", "F6", scale_factor=0.6)
        self.play(FadeIn(g2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Highlight the three sub-triangles
        self.play(g2[1].animate.set_stroke(color="#00FF00", opacity=0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Determine anchor center based on area
        # Fix issue 32: Use B1-F6 area to prevent obstruction
        dummy = Mobject()
        self.place_in_area(dummy, "B1", "F6", scale_factor=0.6)
        main_center = dummy.get_center()

        # Construct Level 3 (27 nodes) by triplicating Level 2
        target_size = 4.0
        h_total = target_size * np.sqrt(3) / 2

        c_top = main_center + np.array([0, h_total/4, 0])
        c_bl = main_center + np.array([-target_size/4, -h_total/4, 0])
        c_br = main_center + np.array([target_size/4, -h_total/4, 0])

        g3_top = g2.copy().scale(0.5).move_to(c_top)
        g3_bl = g2.copy().scale(0.5).move_to(c_bl)
        g3_br = g2.copy().scale(0.5).move_to(c_br)

        # New bridge edges for Level 3 in #FFD700
        bridge_edges = VGroup(
            Line(g3_top[0][1].get_center(), g3_bl[0][0].get_center(), color="#FFD700", stroke_width=3),
            Line(g3_top[0][2].get_center(), g3_br[0][0].get_center(), color="#FFD700", stroke_width=3),
            Line(g3_bl[0][2].get_center(), g3_br[0][1].get_center(), color="#FFD700", stroke_width=3)
        )

        self.play(
            ReplacementTransform(g2, VGroup(g3_top, g3_bl, g3_br)),
            run_time=1.5
        )
        self.play(Create(bridge_edges))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Corner Labels in #FF00FF
        label_000 = Text("000", font_size=18, color="#FF00FF")
        label_111 = Text("111", font_size=18, color="#FF00FF")
        label_222 = Text("222", font_size=18, color="#FF00FF")
        
        # Position labels near corners
        label_000.next_to(g3_top[0][0], UP, buff=0.1)
        label_111.next_to(g3_bl[0][1], LEFT, buff=0.1)
        label_222.next_to(g3_br[0][2], RIGHT, buff=0.1)
        
        self.play(Write(label_000), Write(label_111), Write(label_222))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Zoom into the top corner group to show self-similarity
        full_view = VGroup(g3_top, g3_bl, g3_br, bridge_edges, label_000, label_111, label_222)
        
        # Simulated Zoom
        self.play(
            full_view.animate.scale(2.5).move_to(main_center - (g3_top[0][0].get_center() - main_center)*1.5),
            run_time=2
        )
        self.wait(1)
        
        # Pan out to reveal full 27-state gasket
        self.play(
            full_view.animate.scale(1/2.5).move_to(main_center),
            run_time=2
        )
        self.wait(2)
