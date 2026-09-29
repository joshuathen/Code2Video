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
            "Product Rule applies to multiplying two functions.",
            "Imagine a rectangle growing in both dimensions.",
            "Area change depends on both width and height.",
            "The rule: derivative is f'g plus fg'.",
            "Total weight gain is the classic example."
        ]
        self.setup_layout("The Product Rule", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        
        # Load asset and add label
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg", color=WHITE)
        self.place_at_grid(icon, 'B3', scale_factor=0.6)
        f_g_text = MathTex("f(x) \\cdot g(x)", color=WHITE).scale(0.8)
        f_g_text.next_to(icon, DOWN)
        
        self.play(FadeIn(icon), Write(f_g_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        rect = Rectangle(width=2, height=1.5, color=BLUE)
        self.place_in_area(rect, 'D2', 'E4', scale_factor=0.8)
        width_label = MathTex("f(x)", color=BLUE).next_to(rect, DOWN)
        height_label = MathTex("g(x)", color=BLUE).next_to(rect, LEFT)
        self.play(Create(rect), Write(width_label), Write(height_label))
        self.play(rect.animate.scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Visualize area change
        arrow_w = Arrow(start=ORIGIN, end=RIGHT*0.5, color=RED).next_to(rect, RIGHT)
        arrow_h = Arrow(start=ORIGIN, end=UP*0.5, color=RED).next_to(rect, UP)
        self.play(GrowArrow(arrow_w), GrowArrow(arrow_h))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF6666"))
        formula = MathTex("(f \\cdot g)' = f'g + fg'", color=WHITE)
        self.place_at_grid(formula, 'B4', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#AAAAAA"))
        puppy_text = Text("Weight = Size * Density", font_size=24, color=WHITE)
        self.place_at_grid(puppy_text, 'F5', scale_factor=0.7)
        self.play(FadeIn(puppy_text))
        self.wait(2)
