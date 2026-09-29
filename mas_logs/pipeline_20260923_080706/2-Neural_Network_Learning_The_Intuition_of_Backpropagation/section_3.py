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
        lines = [
            "Think of the network as connected gears.",
            "The chain rule propagates error signals backward.",
            "Error flows like water through a pipe.",
            "Weights adjust themselves to reduce total error.",
            "Backpropagation efficiently calculates these weight updates."
        ]
        self.setup_layout("The Core: Intuitive Backpropagation (The Chain Rule)", lines)
        
        colors = [BLUE_A, YELLOW_A, RED_A, GREEN_A, PURPLE_A]

        # Objects
        gear1 = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
        gear2 = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
        self.place_at_grid(gear1, 'B3', scale_factor=0.6)
        self.place_at_grid(gear2, 'B5', scale_factor=0.6)
        gears = VGroup(gear1, gear2)

        pipe = Line(self.grid['D5'], self.grid['D1'], color=GRAY, stroke_width=20)
        error_dot = Dot(color=RED).move_to(self.grid['D5'])
        label_dw = Text("Δw", font_size=24, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.play(Create(gears))
        self.play(Rotate(gear1, angle=PI), Rotate(gear2, angle=-PI))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        arrow = Arrow(self.grid['B5'], self.grid['B3'], color=YELLOW)
        self.play(GrowArrow(arrow))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        self.play(Create(pipe))
        self.play(error_dot.animate.move_to(self.grid['D1']), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        self.place_at_grid(label_dw, 'A3', scale_factor=0.5)
        self.play(FadeIn(label_dw))
        self.play(gear2.animate.set_color(GREEN))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        formula = MathTex(r"\frac{\partial E}{\partial w} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial w}", font_size=24)
        self.place_in_area(formula, 'D3', 'D5', scale_factor=0.5)
        self.play(Write(formula))
        self.wait(2)
