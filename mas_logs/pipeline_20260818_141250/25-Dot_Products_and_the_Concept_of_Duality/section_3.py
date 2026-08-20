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
        lecture_lines = [
            "Duality connects vectors to linear functionals.",
            "The dot product acts as a bridge.",
            "Every functional corresponds to a vector.",
            "Vectors act as measuring devices.",
            "Duality merges object and operation."
        ]
        self.setup_layout("The Core Concept: Duality", lecture_lines)
        
        # Mobjects
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=RED)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color="#FF00FF")
        
        v_space = Circle(radius=0.4, color=BLUE).add(Text("V", font_size=16, color=BLUE))
        scalar_space = Circle(radius=0.4, color=RED).add(Text("R", font_size=16, color=RED))
        arrow = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        dual_label = Text("Dual Space", font_size=24, color=WHITE)
        self.place_at_grid(dual_label, 'C2', scale_factor=0.8)
        self.place_at_grid(compass, 'C1', scale_factor=0.4)
        self.play(Write(dual_label), FadeIn(compass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(v_space, 'D2')
        self.place_at_grid(scalar_space, 'D5')
        self.place_in_area(arrow, 'D3', 'D4', scale_factor=0.7)
        self.play(Create(v_space), Create(scalar_space), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        dot_product_label = MathTex(r"f(v) = w \cdot v", color=YELLOW)
        self.place_at_grid(dot_product_label, 'C5', scale_factor=0.8)
        self.play(Write(dot_product_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        sensor_box = Square(side_length=0.4, color=GREEN)
        self.place_at_grid(sensor_box, 'E4', scale_factor=0.6)
        self.play(FadeIn(sensor_box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.place_at_grid(ruler, 'F4', scale_factor=0.4)
        self.play(Indicate(v_space), Indicate(scalar_space), FadeIn(ruler))
        self.wait(2)
