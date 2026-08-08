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
        self.setup_layout("The Concept of Dual Graphs", [
            "Dual graphs simplify complex graph structures.",
            "Place a vertex in every graph face.",
            "Connect vertices if original faces share an edge."
        ])
        
        # Original Graph
        v1 = Dot(color=BLUE)
        v2 = Dot(color=BLUE)
        v3 = Dot(color=BLUE)
        v4 = Dot(color=BLUE)
        
        self.place_at_grid(v1, "B3", 1.0)
        self.place_at_grid(v2, "D3", 1.0)
        self.place_at_grid(v3, "C5", 1.0)
        self.place_at_grid(v4, "C4", 1.0)
        
        edges = VGroup(
            Line(v1.get_center(), v2.get_center(), color=GRAY),
            Line(v2.get_center(), v3.get_center(), color=GRAY),
            Line(v3.get_center(), v1.get_center(), color=GRAY),
            Line(v4.get_center(), v1.get_center(), color=GRAY),
            Line(v4.get_center(), v2.get_center(), color=GRAY),
            Line(v4.get_center(), v3.get_center(), color=GRAY)
        )
        
        full_graph = VGroup(v1, v2, v3, v4, edges)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(full_graph))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Place dots in faces - Asset placeholder used as a subtle icon
        # Placeholder asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=RED).scale(0.2)
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=RED).scale(0.2)
        icon3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=RED).scale(0.2)
        
        d1 = Dot(color=RED).move_to(self.grid["C3"])
        d2 = Dot(color=RED).move_to(self.grid["B4"])
        d3 = Dot(color=RED).move_to(self.grid["D4"])
        
        self.place_at_grid(icon1, "C3", 0.5)
        self.place_at_grid(icon2, "B4", 0.5)
        self.place_at_grid(icon3, "D4", 0.5)
        
        dual_graph_group = VGroup(d1, d2, d3, icon1, icon2, icon3)
        self.play(FadeIn(dual_graph_group))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        dual_edges = VGroup(
            DashedLine(d1.get_center(), d2.get_center(), color=WHITE),
            DashedLine(d2.get_center(), d3.get_center(), color=WHITE),
            DashedLine(d3.get_center(), d1.get_center(), color=WHITE)
        )
        
        # Highlight and place in area B3-E5
        dual_full = VGroup(dual_graph_group, dual_edges)
        self.play(Create(dual_edges))
        self.play(dual_full.animate.scale(0.9).move_to(self.grid["C4"])) 
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
