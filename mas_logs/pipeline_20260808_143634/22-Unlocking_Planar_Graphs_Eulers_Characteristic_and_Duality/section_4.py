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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Duality Connection", [
            "Original G's faces become Dual G*'s vertices.",
            "Original G's vertices become Dual G*'s faces.",
            "Duality preserves Euler's structural relationship.",
            "Square: 4V, 4E, 2F morphs correctly.",
            "Duality maintains V - E + F = 2."
        ])
        
        # Create Graph G (Square)
        vertices_g = [UP+LEFT, UP+RIGHT, DOWN+RIGHT, DOWN+LEFT]
        edges_g = [(0, 1), (1, 2), (2, 3), (3, 0)]
        g = Graph(list(range(4)), edges_g, layout={i: v for i, v in enumerate(vertices_g)}, vertex_config={"radius": 0.1, "color": WHITE}, edge_config={"stroke_width": 4, "color": WHITE})
        
        # Create Dual G*
        g_dual = Graph([0, 1], [(0, 1)], layout={0: ORIGIN, 1: UP*2+RIGHT*2}, vertex_config={"radius": 0.1, "color": "#FFA500"}, edge_config={"stroke_width": 4, "color": "#FFA500"})
        
        objs = VGroup(g, g_dual)
        # Fix 30: place_in_area adjustment
        self.place_in_area(objs, "B3", "E5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        # Using placeholder asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        self.play(FadeIn(g), FadeIn(g_dual))
        self.lecture[0].set_color("#FFA500")

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFA500")
        self.play(Indicate(g))

        # === Animation for Lecture Line 3 ===
        # Highlight E and E* with dashed line (referenced in storyboard)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        dashed_line = DashedLine(start=LEFT*2, end=RIGHT*2, color=YELLOW)
        self.place_at_grid(dashed_line, "E2")
        self.play(Create(dashed_line))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FFFF")
        label = Text("V=4, E=4, F=2", font_size=20, color="#00FFFF")
        # Fix 31: place_at_grid adjustment
        self.place_at_grid(label, "F1", scale_factor=0.7)
        self.play(Write(label))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FFFF")
        self.play(FadeOut(label))
        eq = MathTex("V - E + F = 2", color="#00FFFF").scale(0.8)
        # Fix 32: place_at_grid adjustment
        self.place_at_grid(eq, "F3", scale_factor=0.8)
        self.play(Write(eq))
        self.wait(2)
