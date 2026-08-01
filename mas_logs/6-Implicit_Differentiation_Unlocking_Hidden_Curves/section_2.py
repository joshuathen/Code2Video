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
        # Initial Setup
        title_text = "Prerequisite: The Chain Rule Secret"
        lecture_lines = [
            "Treat y as an invisible function of x.",
            "Derivative of y-cubed isn't just three y-squared.",
            "We must multiply by the inner derivative, dy dx."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line in TEAL (standard Manim Cyan equivalent)
        self.play(self.lecture[0].animate.set_color(TEAL))
        
        # Display the chain rule formula as Text to avoid latex dependency error
        formula1 = Text(
            r"d/dx [f(x)^3] = 3f(x)^2 * f'(x)", 
            color=TEAL,
            font_size=24
        )
        self.place_in_area(formula1, "A1", "C6", scale_factor=0.9)
        self.play(Write(formula1))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line focus to the second line (Yellow)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Transform the formula into d/dx [y^3] = 3y^2 * (dy/dx)
        formula2 = Text(
            "d/dx [y^3] = 3y^2 * dy/dx", 
            color=YELLOW,
            font_size=24
        )
        self.place_in_area(formula2, "A1", "C6", scale_factor=0.9)
        
        self.play(Transform(formula1, formula2))
        self.add(formula2)
        self.remove(formula1)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line focus to the third line (Red)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(RED)
        )
        
        # Highlight (dy/dx) in bright red (#FF0000)
        self.play(formula2.animate.set_color("#FF0000"))
        
        # Label it 'Inner Derivative Footprint'
        label = Text("Inner Derivative Footprint", font_size=24, color="#FF0000")
        self.place_at_grid(label, "E4", scale_factor=0.8)
        
        # Point an arrow from the label to the highlighted dy/dx
        arrow = Arrow(
            start=self.grid["E4"] + UP * 0.3, 
            end=formula2.get_bottom() + DOWN * 0.1, 
            color="#FF0000", 
            buff=0.1,
            stroke_width=4
        )
        
        self.play(
            Create(arrow),
            FadeIn(label)
        )
        self.wait(3)
