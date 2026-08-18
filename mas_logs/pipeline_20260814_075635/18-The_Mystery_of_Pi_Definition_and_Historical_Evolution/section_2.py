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
        lecture_lines = [
            "Pi is circumference divided by diameter.",
            "We write this as C equals Pi times d.",
            "Pi is an irrational number.",
            "It never ends and never repeats.",
            "It is a fundamental mathematical constant."
        ]
        self.setup_layout("Defining Pi (π)", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        tape = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tape.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Create elements
        circle = Circle(radius=1, color=WHITE)
        diameter = Line(circle.get_left(), circle.get_right(), color="#00FFFF")
        d_label = MathTex("d", color="#00FFFF").next_to(diameter, UP, buff=0.1)
        
        c_group = VGroup(circle, diameter, d_label, compass)
        self.place_in_area(c_group, 'C2', 'D4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Create(circle), Create(diameter), Write(d_label), FadeIn(compass))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        c_val = MathTex("C = \\pi \\cdot d", color="#FF4500")
        self.place_at_grid(c_val, 'E4', scale_factor=1.0)
        
        c_arc = Arc(radius=1, start_angle=0, angle=2*PI, color="#00FF00")
        self.place_at_grid(c_arc, 'C3', scale_factor=0.8)
        
        self.play(Create(c_arc), Write(c_val), FadeIn(tape))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        ratio = MathTex("\\frac{C}{d} = \\pi", color="#FF4500")
        self.place_at_grid(ratio, 'A4', scale_factor=1.2)
        self.play(Write(ratio))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        pi_digits = Text("3.14159...", font_size=36, color="#FF4500")
        self.place_at_grid(pi_digits, 'B6', scale_factor=0.8)
        self.play(Write(pi_digits))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.place_at_grid(ruler, 'D6', scale_factor=0.8)
        self.play(FadeIn(ruler))
        self.wait(2)
