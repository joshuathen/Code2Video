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
        self.setup_layout("Core Example: The Quadratic Map f(z) = z² + c", [
            "Consider the quadratic map.",
            "The constant C changes dynamics.",
            "Varying C transforms Julia sets.",
            "Simple rules create infinite complexity.",
            "The Mandelbrot set maps behavior."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        func_label = MathTex("f(z) = z^2 + c", color=WHITE)
        self.place_in_area(func_label, 'A2', 'B5', scale_factor=1.2)
        self.play(Write(func_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(RED))
        c_label = MathTex("c", color=RED)
        self.place_at_grid(c_label, "D2")
        self.play(Write(c_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        # Simple geometric representation of set
        julia_shape = Circle(radius=1.0, color=YELLOW)
        self.place_in_area(julia_shape, 'C3', 'E5', scale_factor=0.8)
        self.play(Create(julia_shape))
        
        # Animate c transition
        c_val = ValueTracker(0)
        self.play(c_val.animate.set_value(-0.8), run_time=2)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        complex_text = Text("Complexity", font_size=24, color=GREEN)
        self.place_at_grid(complex_text, 'E2', scale_factor=0.9)
        self.play(Write(complex_text))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(PURPLE))
        mandel_text = Text("Mandelbrot", font_size=24, color=PURPLE)
        self.place_at_grid(mandel_text, 'E5', scale_factor=0.9)
        self.play(Write(mandel_text))
        
        self.wait(2)
