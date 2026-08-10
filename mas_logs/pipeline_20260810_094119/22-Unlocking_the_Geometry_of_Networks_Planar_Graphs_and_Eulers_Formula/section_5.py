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
        self.setup_layout("Real-world Application & Summary", [
            "Duality helps optimize circuit design.",
            "Avoid short-circuits by rerouting.",
            "Simplify complex systems into geometry."
        ])
        
        # Load asset - SVGMobject is preferred for .svg files
        map_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color=WHITE)
        dual_graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color="#33FF57")
        highlight_edge = Line(start=LEFT, end=RIGHT, color=RED).scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(map_obj, 'B3', 'C4', scale_factor=0.6)
        self.play(Create(map_obj))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.place_in_area(dual_graph, 'E3', 'F5', scale_factor=0.7)
        self.play(FadeIn(dual_graph))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # Highlight part
        self.place_at_grid(highlight_edge, 'D3', scale_factor=0.8)
        self.play(Indicate(highlight_edge))
        
        # Formula
        formula = MathTex(r"V - E + F = 2", color=YELLOW).scale(1.0)
        self.place_at_grid(formula, 'D4', scale_factor=1.2)
        self.play(Write(formula))
        self.wait(2)
