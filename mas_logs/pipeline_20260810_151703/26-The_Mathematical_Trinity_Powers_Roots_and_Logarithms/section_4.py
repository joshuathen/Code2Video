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
        self.setup_layout("Unifying the Notation (The Triple Triangle)", [
            "These operations form a perfect triangle.",
            "Three values rotate through three different roles.",
            "Two cubed is eight, log base two is three."
        ])
        
        # Load asset
        cube_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg").scale(0.5)
        
        # Define vertices labels
        b = MathTex(r"b", color=WHITE)
        x = MathTex(r"x", color=WHITE)
        y = MathTex(r"y", color=WHITE)
        
        # Labels positioning (Applied fixes from issues 30, 45)
        self.place_at_grid(b, "B3", scale_factor=1.0)
        self.place_at_grid(x, "D2", scale_factor=1.0)
        self.place_at_grid(y, "D4", scale_factor=1.0)
        
        # Triangle group (Applied fix from issues 29, 44)
        triangle = Polygon(self.grid["B3"], self.grid["D2"], self.grid["D4"], color=WHITE)
        triangle_group = VGroup(triangle, b, x, y, cube_icon)
        self.place_in_area(triangle_group, "C2", "E4", scale_factor=0.9)
        
        # Edges
        edge_1 = MathTex(r"b^x = y", color=WHITE)
        edge_2 = MathTex(r"\log_b(y) = x", color=WHITE)
        edge_3 = MathTex(r"y^{1/x} = b", color=WHITE)
        
        # Formula positioning (Applied fix from issues 31, 46)
        main_formula = VGroup(edge_1, edge_2, edge_3).arrange(DOWN)
        self.place_in_area(main_formula, "E2", "F4", scale_factor=0.85)
        
        self.play(Create(triangle))
        self.play(FadeIn(b), FadeIn(x), FadeIn(y), FadeIn(cube_icon))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(edge_1), FadeIn(edge_2), FadeIn(edge_3))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(
            edge_1.animate.set_color("#FF0000"),
            edge_2.animate.set_color("#00FF00"),
            edge_3.animate.set_color("#0000FF"),
        )
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Rotate(triangle_group, angle=PI/3, about_point=self.grid["C3"]))
        self.wait(1)
