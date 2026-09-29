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
        self.setup_layout("The Scaling Law: Prerequisite Knowledge", [
            "Scaling a shape by factor s yields N copies.", 
            "The relationship is N equals s to power D.", 
            "Solving for D gives the log ratio formula."
        ])
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        line = Line(LEFT*1.5, RIGHT*1.5, color=BLUE)
        self.place_at_grid(line, 'C2', scale_factor=0.8)
        self.play(Create(line))
        self.place_at_grid(ruler, 'B3', scale_factor=0.3)
        self.play(FadeIn(ruler))
        
        segments = VGroup(*[Line(ORIGIN, RIGHT*1, color=YELLOW) for _ in range(3)]).arrange(RIGHT, buff=0)
        self.place_at_grid(segments, 'C3', scale_factor=0.8)
        self.play(Transform(line.copy(), segments))
        
        eq1 = MathTex(r"3^1 = 3", color=YELLOW)
        self.place_at_grid(eq1, 'D3')
        self.play(Write(eq1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        square = Square(side_length=1.5, color=BLUE)
        self.place_at_grid(square, 'B4', scale_factor=0.7)
        self.play(Create(square))
        self.place_at_grid(blocks, 'A4', scale_factor=0.3)
        self.play(FadeIn(blocks))
        
        small_squares = VGroup(*[Square(side_length=0.5, color=YELLOW) for _ in range(9)]).arrange_in_grid(3, 3, buff=0)
        self.place_at_grid(small_squares, 'C4', scale_factor=0.7)
        self.play(Transform(square.copy(), small_squares))
        
        eq2 = MathTex(r"3^2 = 9", color=YELLOW)
        self.place_at_grid(eq2, 'D4')
        self.play(Write(eq2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        formula = MathTex(r"N = s^D \implies D = \frac{\log(N)}{\log(s)}", color=WHITE)
        self.place_in_area(formula, 'D3', 'E5', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
