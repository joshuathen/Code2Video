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
        lecture_lines = [
            "Dot product bridges vectors and functionals.",
            "It is foundational in physics and optimization.",
            "It defines how systems sense inputs."
        ]
        self.setup_layout("Application: Duality in Action", lecture_lines)
        
        # Elements
        vector_v = Vector([1.5, 1], color=BLUE)
        functional_f = Rectangle(height=0.5, width=2, color=RED, fill_opacity=0.3)
        label_v = Text("Vector", font_size=24, color=BLUE)
        label_f = Text("Functional", font_size=24, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(vector_v, 'B2', scale_factor=0.7)
        self.place_at_grid(label_v, 'B3', scale_factor=0.6)
        self.place_at_grid(functional_f, 'D2', scale_factor=0.7)
        self.place_at_grid(label_f, 'D3', scale_factor=0.6)
        self.play(Create(vector_v), Write(label_v), Create(functional_f), Write(label_f))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        physics_icon = Circle(radius=0.5, color=GREEN)
        opt_icon = Square(side_length=0.8, color=YELLOW)
        self.place_at_grid(physics_icon, 'B5', scale_factor=0.5)
        self.place_at_grid(opt_icon, 'D5', scale_factor=0.5)
        self.play(DrawBorderThenFill(physics_icon), DrawBorderThenFill(opt_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        sensor = Dot(color=WHITE)
        self.place_at_grid(sensor, 'C5', scale_factor=0.5)
        self.play(FadeIn(sensor))
        self.play(sensor.animate.move_to(self.grid['B2']), run_time=1.5)
        self.play(sensor.animate.move_to(self.grid['D2']), run_time=1.5)
