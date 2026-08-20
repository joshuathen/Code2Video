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
        self.setup_layout("The Mathematical DNA: The Fourier Formula", [
            "The Fourier series models periodic functions as sums.", 
            "Coefficients act as volume knobs for frequencies.", 
            "Adding more sine waves refines the sharp edges.", 
            "Each term builds the target waveform piece-by-piece.", 
            "The sum creates complex, precise mathematical shapes."
        ])
        
        knob_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"e^{i\theta} = \cos(\theta) + i\sin(\theta)", color=WHITE)
        self.place_at_grid(formula, 'B1', scale_factor=0.8)
        knob1 = SVGMobject(knob_path).scale(0.3).next_to(formula, RIGHT)
        self.play(Write(formula), FadeIn(knob1))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        dot = Dot(color=YELLOW)
        circle = Circle(radius=0.7, color=WHITE).shift(self.grid["E3"])
        dot.move_to(circle.get_right())
        self.play(Create(circle), FadeIn(dot))
        self.play(Rotate(dot, angle=2*PI, about_point=circle.get_center()), run_time=2)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        projection = DashedLine(dot.get_center(), [dot.get_x(), circle.get_center()[1], 0], color=RED)
        self.play(Create(projection))
        self.lecture[2].set_color(RED)

        # === Animation for Lecture Line 4 ===
        sum_text = MathTex(r"f(x) = \sum_{n} c_n \cdot \text{knob}_n", font_size=24, color=GREEN)
        self.place_at_grid(sum_text, 'C2', scale_factor=0.7)
        self.play(FadeIn(sum_text))
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        integral = MathTex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos(nx) + b_n \sin(nx))", color="#00FF00")
        self.place_in_area(integral, 'D2', 'D5', scale_factor=0.6)
        knob2 = SVGMobject(knob_path).scale(0.3).next_to(integral, DOWN)
        self.play(Write(integral), FadeIn(knob2))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
