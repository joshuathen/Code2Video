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
        self.setup_layout(
            "The Bridge: Connecting the Dual", 
            [
                "- Connect dual vertices where original faces share an edge.",
                "- One dual edge crosses exactly one original edge.",
                "- The number of edges remains exactly the same."
            ]
        )

        # Graph setup (Planar graph with V=4, E=6, F=4)
        v1 = self.grid["B3"] + UP * 0.5
        v2 = self.grid["E2"]
        v3 = self.grid["E4"]
        v4 = self.grid["D3"] # Inner vertex

        edges_config = [
            (v1, v2), (v2, v3), (v3, v1), # Outer triangle
            (v1, v4), (v2, v4), (v3, v4)  # Inner edges
        ]
        
        original_edges = VGroup(*[Line(start, end, color="#BBBBBB", stroke_width=3) for start, end in edges_config])
        
        # Dual Vertices (Face centers)
        d1_pos = (v1 + v2 + v4) / 3
        d2_pos = (v2 + v3 + v4) / 3
        d3_pos = (v3 + v1 + v4) / 3
        dout_pos = self.grid["A5"]

        dual_vertices_pos = [d1_pos, d2_pos, d3_pos, dout_pos]
        dual_vertices = VGroup(*[Dot(pos, color="#FFA500", radius=0.08) for pos in dual_vertices_pos])
        
        # Dual Edges
        dual_edges_config = [
            (d1_pos, dout_pos), # crosses (v1, v2) -> label 1
            (d2_pos, dout_pos), # crosses (v2, v3) -> label 2
            (d3_pos, dout_pos), # crosses (v3, v1) -> label 3
            (d1_pos, d3_pos),   # crosses (v1, v4) -> label 4
            (d1_pos, d2_pos),   # crosses (v2, v4) -> label 5
            (d2_pos, d3_pos)    # crosses (v3, v4) -> label 6
        ]
        
        dual_edges = VGroup(*[Line(start, end, color="#FFA500", stroke_width=4) for start, end in dual_edges_config])

        # Assets
        fence_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fence.svg")
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")

        # Initial display
        self.add(original_edges)
        self.add(dual_vertices)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        self.play(Create(dual_edges), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )

        # Highlight one 'fence' (index 4: v2-v4) and its 'bridge' (index 4: d1-d2)
        target_fence = original_edges[4]
        target_bridge = dual_edges[4]
        
        fence_icon.scale(0.3).move_to(target_fence.get_center()).set_color("#FFFF00")
        bridge_icon.scale(0.3).move_to(target_bridge.get_center()).set_color("#FFA500")

        self.play(
            target_fence.animate.set_color("#FFFF00").set_stroke(width=6),
            FadeIn(fence_icon, shift=UP*0.2),
            target_bridge.animate.set_stroke(width=8),
            FadeIn(bridge_icon, shift=DOWN*0.2)
        )
        self.wait(2)
        
        self.play(
            target_fence.animate.set_color("#BBBBBB").set_stroke(width=3),
            FadeOut(fence_icon),
            target_bridge.animate.set_stroke(width=4),
            FadeOut(bridge_icon)
        )

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFA500")
        )

        # Create labels for counting
        label_1 = VGroup(Dot(color=WHITE, radius=0.2), Text("1", font_size=20, color=BLACK))
        self.place_at_grid(label_1, 'B5', scale_factor=0.5)

        label_2 = VGroup(Dot(color=WHITE, radius=0.2), Text("2", font_size=20, color=BLACK))
        self.place_at_grid(label_2, 'C6', scale_factor=0.5)

        label_3 = VGroup(Dot(color=WHITE, radius=0.2), Text("3", font_size=20, color=BLACK))
        self.place_at_grid(label_3, 'B6', scale_factor=0.5)

        label_4 = VGroup(Dot(color=WHITE, radius=0.2), Text("4", font_size=20, color=BLACK))
        self.place_at_grid(label_4, 'C4', scale_factor=0.5)

        label_5 = VGroup(Dot(color=WHITE, radius=0.2), Text("5", font_size=20, color=BLACK))
        self.place_at_grid(label_5, 'E4', scale_factor=0.5)

        label_6 = VGroup(Dot(color=WHITE, radius=0.2), Text("6", font_size=20, color=BLACK))
        self.place_at_grid(label_6, 'E5', scale_factor=0.5)

        labels = [label_1, label_2, label_3, label_4, label_5, label_6]
        for i, lbl in enumerate(labels):
            lbl[1].move_to(lbl[0].get_center())
            self.play(
                dual_edges[i].animate.set_color(WHITE),
                FadeIn(lbl),
                run_time=0.4
            )
            self.play(
                dual_edges[i].animate.set_color("#FFA500"),
                run_time=0.1
            )

        # Formula positioning (Issue 23)
        e_star_text = MathTex("E^* = 6", color="#FFA500", font_size=36)
        self.place_in_area(e_star_text, 'F4', 'F6', scale_factor=1.0)
        
        self.play(Write(e_star_text))
        self.wait(2)

        # Cleanup for next section
        self.play(
            *[FadeOut(lbl) for lbl in labels],
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
