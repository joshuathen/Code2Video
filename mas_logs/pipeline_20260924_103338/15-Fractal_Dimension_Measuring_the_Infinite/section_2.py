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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Scaling Laws", [
            "Scaling determines how an object fills space.",
            "Line segments scale with the power of one.",
            "Squares scale with the power of two."
        ])
        
        c1 = "#00FFFF" # Cyan
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        tile = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tile.svg")
        
        # === Animation for Lecture Line 1 ===
        # Scaling determines how an object fills space.
        self.lecture[0].set_color(c1)
        line = Line(LEFT, RIGHT, color=c1)
        self.place_in_area(line, "A2", "B3", scale_factor=0.6)
        
        # Ruler asset
        ruler_copy = ruler.copy()
        self.place_at_grid(ruler_copy, "B5", scale_factor=0.4)
        
        self.play(Create(line), FadeIn(ruler_copy))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line segments scale with the power of one.
        self.lecture[1].set_color(c1)
        line_scaled = VGroup(
            Line(LEFT*0.5, ORIGIN, color=c1),
            Line(ORIGIN, RIGHT*0.5, color=c1)
        ).arrange(RIGHT, buff=0)
        self.place_in_area(line_scaled, "A4", "B5", scale_factor=0.6)
        
        self.play(Transform(line, line_scaled))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Squares scale with the power of two.
        self.lecture[2].set_color(c1)
        
        square = Square(side_length=1.5, color=c1, fill_opacity=0.3)
        self.place_at_grid(square, "D2", scale_factor=0.6)
        
        four_squares = VGroup(*[Square(side_length=0.75, color=c1, fill_opacity=0.5) for _ in range(4)])
        four_squares.arrange_in_grid(2, 2, buff=0)
        self.place_at_grid(four_squares, "D4", scale_factor=0.6)
        
        # Tile asset
        tile_copy = tile.copy()
        self.place_at_grid(tile_copy, "E6", scale_factor=0.4)
        
        relation = MathTex("N=S^D", color=WHITE)
        self.place_at_grid(relation, "F4", scale_factor=0.8)
        
        self.play(Create(square), FadeIn(tile_copy))
        self.play(Transform(square, four_squares), Write(relation))
        self.wait(2)
