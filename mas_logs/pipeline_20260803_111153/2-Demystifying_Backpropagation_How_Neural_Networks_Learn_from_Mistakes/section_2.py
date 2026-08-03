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
        # Setup title and lecture
        title_text = "The Cost Function: Measuring the 'Ouch'"
        lecture_lines = [
            "We measure how far the prediction is from reality.",
            "This difference is called the loss or cost.",
            "Squaring the error ensures it is always positive."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_TARGET = "#00FF00"
        COLOR_PREDICTION = "#FF0000"
        COLOR_ERROR = "#FF0000"
        COLOR_PARABOLA = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # Number line
        number_line = NumberLine(
            x_range=[0, 10, 1],
            length=5,
            include_numbers=True,
            font_size=18,
            color=WHITE
        )
        self.place_in_area(number_line, "C1", "C6")

        target_val = 4.0
        prediction_val = 9.2

        target_dot = Dot(number_line.n2p(target_val), color=COLOR_TARGET)
        target_label = Text(f"Reality: {target_val}", font_size=16, color=COLOR_TARGET).next_to(target_dot, DOWN)

        pred_dot = Dot(number_line.n2p(prediction_val), color=COLOR_PREDICTION)
        pred_label = Text(f"Prediction: {prediction_val}", font_size=16, color=COLOR_PREDICTION).next_to(pred_dot, DOWN)

        self.play(Create(number_line))
        self.play(FadeIn(target_dot, target_label), FadeIn(pred_dot, pred_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        error_brace = BraceBetweenPoints(target_dot.get_center(), pred_dot.get_center(), UP, color=COLOR_ERROR)
        error_text = Text("Error", font_size=20, color=COLOR_ERROR).next_to(error_brace, UP)

        self.play(Create(error_brace), Write(error_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Transform to axes and parabola
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": False},
            tips=False
        )
        self.place_in_area(axes, "B1", "E6")

        parabola = axes.plot(lambda x: x**2, x_range=[-3, 3], color=COLOR_PARABOLA)
        parabola_eq = MathTex(r"\text{Cost} = (\text{Error})^2", font_size=24, color=COLOR_PARABOLA)
        self.place_at_grid(parabola_eq, "A5")

        # Represent error as x-coord on parabola
        error_val = 2.0
        parabola_dot = Dot(axes.c2p(error_val, error_val**2), color=COLOR_ERROR)
        parabola_dot_label = Text("Positive Cost", font_size=18, color=COLOR_ERROR).next_to(parabola_dot, UR, buff=0.1)

        self.play(
            FadeOut(number_line, target_dot, target_label, pred_dot, pred_label, error_brace, error_text),
            Create(axes),
            Create(parabola),
            Write(parabola_eq)
        )
        self.play(FadeIn(parabola_dot, parabola_dot_label))
        self.wait(3)
        
        self.lecture[2].set_color(WHITE)