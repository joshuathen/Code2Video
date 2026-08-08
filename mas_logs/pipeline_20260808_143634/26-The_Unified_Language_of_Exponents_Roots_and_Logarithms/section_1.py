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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Powers show repeated multiplication growth.",
            "Base b raised to x equals y.",
            "Example: 2 raised to 3 equals 8.",
            "This is the foundation of growth.",
            "Base, exponent, and power are linked."
        ]
        self.setup_layout("The Foundation: Powers as Growth", lecture_lines)
        
        # Load Assets
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg").scale(0.5)
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg").scale(0.5)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg").scale(0.5)
        counter = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/counter.svg").scale(0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        base = MathTex("b", color=WHITE).scale(1.5)
        self.place_at_grid(base, 'B3', scale_factor=0.6)
        self.place_at_grid(abacus, 'A3')
        self.play(Write(base), FadeIn(abacus))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF6347")
        exponent = MathTex("x", color="#FF6347").scale(1.0).next_to(base, UP, buff=0.1)
        self.play(Write(exponent))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        calc_text = MathTex("2^3 = 8", color="#FFFF00").scale(1.2)
        self.place_at_grid(calc_text, 'C4')
        self.place_at_grid(calculator, 'C5')
        self.play(Write(calc_text), FadeIn(calculator))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#9370DB")
        curve = FunctionGraph(lambda t: 0.1 * (2**(t*2)), x_range=[-1, 2]).set_color("#9370DB")
        self.place_in_area(curve, 'D4', 'F6', scale_factor=0.5)
        self.place_at_grid(ruler, 'D6')
        self.play(Create(curve), FadeIn(ruler))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#32CD32")
        frame = SurroundingRectangle(VGroup(base, exponent), color="#32CD32")
        self.place_at_grid(counter, 'B5')
        self.play(Create(frame), FadeIn(counter))
        self.wait(2)
