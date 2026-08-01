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
        self.setup_layout("The Mathematical Engine: Summation and Activation", [
            "First, we multiply every input by its weight.",
            "Next, we sum these products and add the bias.",
            "This total enters a gate called an activation function.",
            "Sigmoid or ReLU functions determine the neuron's final output.",
            "This math decides if a signal is strong enough."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Line 1: "First, we multiply every input by its weight."
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        # Show partial products: x * w
        partial_eq = MathTex(r"x \cdot w", color="#FFFFFF")
        # Fix cramped formulas: Place in area 'B3'-'B5' (scale 1.1) instead of a single point (Issue 28, 41.1a)
        self.place_in_area(partial_eq, 'B3', 'B5', scale_factor=1.1)
        self.play(Write(partial_eq))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Next, we sum these products and add the bias."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Full equation: z = Σ(x * w) + b
        # Splitting parts to allow targeting Σ for the morph later
        full_equation = MathTex(r"z", "=", r"\sum", "(x \cdot w)", "+", "b", color="#FFFFFF")
        # Ensure it starts at the same position as partial_eq and matches its scale (1.1)
        full_equation.scale(1.1).move_to(partial_eq.get_center())
        
        self.play(ReplacementTransform(partial_eq, full_equation))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Line 3: "This total enters a gate called an activation function."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Target the Σ symbol (index 2)
        sigma = full_equation[2]
        self.play(Indicate(sigma, color="#FF00FF"))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Line 4: "Sigmoid or ReLU functions determine the neuron's final output."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        
        # Setup Axes for the activation function curve
        axes = Axes(
            x_range=[-4, 4, 2],
            y_range=[0, 1, 0.5],
            x_length=4.5,
            y_length=3,
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.9)
        
        # Sigmoid curve and its formula
        sigmoid_curve = axes.plot(lambda x: 1 / (1 + np.exp(-x)), color="#FF00FF")
        sigmoid_label = MathTex(r"\sigma(z) = \frac{1}{1 + e^{-z}}", color="#FF00FF")
        # Fix illegible formula: Place in area 'B3'-'B5' (scale 0.8) (Issue 29, 41.1b)
        self.place_in_area(sigmoid_label, 'B3', 'B5', scale_factor=0.8)

        # Identify other parts of the equation to fade out
        others = VGroup(*[full_equation[i] for i in range(len(full_equation)) if i != 2])
        
        # Morphing transition
        self.play(
            FadeOut(others),
            ReplacementTransform(sigma, sigmoid_curve),
            Create(axes),
            Write(sigmoid_label)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Line 5: "This math decides if a signal is strong enough."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFF00")
        )
        
        # Highlighting the 'Active' region of the Sigmoid curve in Green (#00FF00)
        active_region = axes.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[0, 4], color="#00FF00")
        active_region.set_stroke(width=6)
        
        active_label = Text("Active Signal", color="#00FF00")
        # Fix disconnected label: Move to 'E5'-'E6' (scale 0.7) (Issue 30, 41.2)
        self.place_in_area(active_label, 'E5', 'E6', scale_factor=0.7)
        
        self.play(
            Create(active_region),
            Write(active_label)
        )
        self.wait(2.0)
