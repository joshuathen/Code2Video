from manim import *
import numpy as np

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
        title_text = "The Engine: Iteration and Functions"
        lecture_lines = [
            "Iteration is the heartbeat of this system.",
            "We apply a mathematical rule repeatedly.",
            "The formula z next equals f of z.",
            "Each output becomes the next step's input.",
            "An orbit tracks a point's journey through time."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow for current line highlight
        COLOR_FORMULA = "#FFFFFF"   # White
        COLOR_DIAGRAM = "#00FFFF"   # Cyan
        COLOR_BOX = "#FF8800"       # Orange

        # === Animation for Lecture Line 1 ===
        # "Iteration is the heartbeat of this system."
        # Display the formula 'f(z) = z^2 + c' centered in white (#FFFFFF).
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        formula_f = MathTex("f(z) = z^2 + c", color=COLOR_FORMULA)
        # Issue 26: Fixed scaling to 1.0
        self.place_in_area(formula_f, "A3", "A4", scale_factor=1.0)
        
        self.play(Write(formula_f))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We apply a mathematical rule repeatedly."
        # Animate a flow diagram: Input z -> Function Box -> Output z_next -> Input z.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        input_z = MathTex("z", color=COLOR_DIAGRAM)
        self.place_at_grid(input_z, "C2")
        
        box = Rectangle(width=1.0, height=0.8, color=COLOR_BOX)
        box_label = MathTex("f", color=COLOR_BOX)
        func_box = VGroup(box, box_label)
        self.place_at_grid(func_box, "C3")
        
        output_z = MathTex("z_{next}", color=COLOR_DIAGRAM)
        # Issue 27: Fixed scaling to 0.8
        self.place_at_grid(output_z, "C4", scale_factor=0.8)
        
        arrow1 = Arrow(input_z.get_right(), box.get_left(), buff=0.1, color=COLOR_DIAGRAM)
        arrow2 = Arrow(box.get_right(), output_z.get_left(), buff=0.1, color=COLOR_DIAGRAM)
        
        self.play(
            FadeIn(input_z),
            Create(arrow1),
            Create(func_box),
            Create(arrow2),
            FadeIn(output_z)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The formula z next equals f of z."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        formula_z_next = MathTex("z_{next} = f(z)", color=COLOR_FORMULA)
        self.place_in_area(formula_z_next, "B3", "B4", scale_factor=1.0)
        
        self.play(FadeIn(formula_z_next))
        self.play(Indicate(formula_z_next))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Each output becomes the next step's input."
        # Feedback loop animation.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        feedback_path = VGroup(
            Line(output_z.get_bottom(), self.grid["D4"], color=COLOR_DIAGRAM),
            Line(self.grid["D4"], self.grid["D2"], color=COLOR_DIAGRAM),
            Arrow(self.grid["D2"], input_z.get_bottom(), buff=0.1, color=COLOR_DIAGRAM)
        )
        
        self.play(Create(feedback_path))
        self.play(Indicate(input_z))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "An orbit tracks a point's journey through time."
        # Show a point at 0 on a number line, then jumping to 1, 2, and 5.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        n_line = NumberLine(x_range=[0, 6, 1], length=4, include_numbers=True, color=WHITE)
        # Issue 25: Fixed area to E2-E6 and scale factor to 0.7
        self.place_in_area(n_line, "E2", "E6", scale_factor=0.7)
        
        dot = Dot(color=COLOR_HIGHLIGHT)
        dot.move_to(n_line.n2p(0))
        
        self.play(Create(n_line))
        self.play(FadeIn(dot))
        
        # Jumps: 0 -> 1 -> 2 -> 5
        jumps = [1, 2, 5]
        for val in jumps:
            self.play(dot.animate.move_to(n_line.n2p(val)), run_time=0.8)
        
        # Pulsing highlight for numbers 1, 2, 5
        # The labels in n_line.numbers correspond to the tick values.
        labels_to_pulse = VGroup(*[n_line.numbers[val] for val in [1, 2, 5]])
        self.play(
            labels_to_pulse.animate.scale(1.5).set_color(COLOR_HIGHLIGHT),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # Accelerate and disappear
        target_pos = n_line.n2p(10) 
        self.play(
            dot.animate.move_to(target_pos).set_rate_func(rate_functions.ease_in_expo),
            run_time=1.5
        )
        self.play(FadeOut(dot))
        
        self.wait(2)
