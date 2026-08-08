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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the scene with the specific title and lecture lines
        self.setup_layout(
            "Advanced Insight: The Self-Dual Mystery and Application",
            [
                "Self-dual graphs transform back into themselves.",
                "The tetrahedron is a classic self-dual example.",
                "Duality aids in circuit design and navigation."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line in #FFFF00
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)

        # Create Graph G (Planar projection of a tetrahedron) in #FF8C00
        v1 = Dot(radius=0.12, color="#FF8C00")
        v2 = Dot(radius=0.12, color="#FF8C00")
        v3 = Dot(radius=0.12, color="#FF8C00")
        v4 = Dot(radius=0.12, color="#FF8C00")
        
        # Internal relative geometry for the projection
        v1.move_to(UP * 1.6)
        v2.move_to(LEFT * 2.0 + DOWN * 1.2)
        v3.move_to(RIGHT * 2.0 + DOWN * 1.2)
        v4.move_to(ORIGIN)
        
        e12 = Line(v1, v2, color="#FF8C00", stroke_width=4)
        e23 = Line(v2, v3, color="#FF8C00", stroke_width=4)
        e31 = Line(v3, v1, color="#FF8C00", stroke_width=4)
        e14 = Line(v1, v4, color="#FF8C00", stroke_width=4)
        e24 = Line(v2, v4, color="#FF8C00", stroke_width=4)
        e34 = Line(v3, v4, color="#FF8C00", stroke_width=4)
        
        graph_g = VGroup(e12, e23, e31, e14, e24, e34, v1, v2, v3, v4)
        # Place graph in central right area (B2 to E5)
        self.place_in_area(graph_g, "B2", "E5", scale_factor=0.6)
        
        # Label G centered at the top of the grid area (Fix Issue 27)
        label_g = MathTex("G", color="#FF8C00")
        self.place_in_area(label_g, 'A3', 'A4', scale_factor=0.8)
        
        self.play(FadeIn(graph_g), Write(label_g))
        self.wait(1)

        # Morph G into its Dual G* showing isomorphism
        graph_g_star = graph_g.copy().set_color("#00FFFF")
        # Use area positioning for G* isomorphism label (Fix Issue 28)
        label_g_star = MathTex("G^* \\cong G", color="#00FFFF")
        self.place_in_area(label_g_star, 'A3', 'A4', scale_factor=0.8)

        self.play(
            ReplacementTransform(graph_g, graph_g_star),
            ReplacementTransform(label_g, label_g_star),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to the second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=0.5
        )

        # Explicit construction of the dual within the faces of G*
        # Retrieve coordinates from the projected vertices
        cv1 = graph_g_star[6].get_center()
        cv2 = graph_g_star[7].get_center()
        cv3 = graph_g_star[8].get_center()
        cv4 = graph_g_star[9].get_center()

        # Place dual nodes in the 3 interior faces and 1 exterior face
        d1 = Dot(radius=0.1, color="#00FFFF").move_to((cv1 + cv2 + cv4) / 3) 
        d2 = Dot(radius=0.1, color="#00FFFF").move_to((cv2 + cv3 + cv4) / 3) 
        d3 = Dot(radius=0.1, color="#00FFFF").move_to((cv3 + cv1 + cv4) / 3) 
        d4 = Dot(radius=0.1, color="#00FFFF").move_to(cv1 + UP * 0.9)        

        self.play(FadeIn(d1, d2, d3, d4))

        # Connect dual nodes across G* edges
        de12 = Line(d1, d2, color="#00FFFF", stroke_width=2)
        de23 = Line(d2, d3, color="#00FFFF", stroke_width=2)
        de31 = Line(d3, d1, color="#00FFFF", stroke_width=2)
        de41 = Line(d4, d1, color="#00FFFF", stroke_width=2)
        de42 = Line(d4, d2, color="#00FFFF", stroke_width=2)
        de43 = Line(d4, d3, color="#00FFFF", stroke_width=2)

        dual_lines = VGroup(de12, de23, de31, de41, de42, de43)
        self.play(Create(dual_lines))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to the third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )

        # Transition the graph structure to a White circuit schematic
        self.play(FadeOut(dual_lines, d1, d2, d3, d4, label_g_star))
        self.play(graph_g_star.animate.set_color(WHITE))
        
        # Add 'Circuit Schematic' label using wide area positioning (Fix Issue 29)
        label_circuit = Text("Circuit Schematic", font_size=24, color=WHITE)
        self.place_in_area(label_circuit, 'A2', 'A5', scale_factor=0.8)
        self.play(FadeIn(label_circuit))

        # Pulse edges in #FFFFFF to represent mesh current flow
        pulses = [
            edge.animate(rate_func=there_and_back).set_stroke(width=10)
            for edge in graph_g_star[:6]
        ]
        self.play(AnimationGroup(*pulses, lag_ratio=0.15), run_time=2.5)
        
        self.wait(2)

        # Final cleanup and reset lecture colors
        self.play(
            FadeOut(graph_g_star, label_circuit),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
