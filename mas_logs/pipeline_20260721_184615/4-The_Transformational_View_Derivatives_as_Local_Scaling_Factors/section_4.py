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

class Section4Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "Formalizing the Derivative as a Density Change"
        lecture_lines = [
            "Equally spaced input dots represent a uniform start.",
            "Sine spreads dots at zero and bunches at pi-half.",
            "The derivative measures the resulting density of the mapping."
        ]
        self.setup_layout(title_text, lecture_lines)

        # --- Visual Objects ---

        # 1. Formulas (Grid Row A)
        # Fallback to segments if MathTex issues occur (L022)
        formula = MathTex(r"f'(x) = \frac{dy}{dx}", color="#FFFFFF")
        density_text = Text("Derivative = Density Change", font_size=24, color="#FFFF00")
        
        # Fix Issue 29: Use place_in_area with scale 0.6 to prevent truncation
        self.place_in_area(formula, "A2", "A3", scale_factor=0.8)
        self.place_in_area(density_text, "A4", "A6", scale_factor=0.6)

        # 2. Graph (Grid Rows B-D)
        # Using Axes as a secondary visual reference
        axes = Axes(
            x_range=[0, PI/2 + 0.3, PI/4],
            y_range=[0, 1.2, 0.5],
            x_length=3.5,
            y_length=2,
            axis_config={"include_tip": True, "color": "#888888", "stroke_width": 2}
        )
        sine_graph = axes.plot(lambda x: np.sin(x), x_range=[0, PI/2], color="#FFFF00")
        graph_label = MathTex(r"f(x) = \sin(x)", color="#FFFF00", font_size=20)
        
        graph_group = VGroup(axes, sine_graph)
        self.place_in_area(graph_group, "B2", "D6", scale_factor=0.8)
        graph_label.next_to(axes, UP, buff=0.1)

        # 3. Lines and Labels (Grid Rows E-F)
        input_line = NumberLine(x_range=[0, PI/2, PI/4], length=4, include_tip=True, color="#87CEEB")
        output_line = NumberLine(x_range=[0, 1.1, 0.25], length=4, include_tip=True, color="#FFD700")
        
        input_label = Text("Input x", font_size=22, color="#87CEEB")
        output_label = Text("Output f(x)", font_size=22, color="#FFD700")

        # Fix Issue 27 and 28: Use place_in_area spanning 2 grid cells for multi-word labels
        self.place_in_area(input_label, "E2", "E3", scale_factor=0.7)
        self.place_in_area(input_line, "E4", "E6", scale_factor=0.9)
        self.place_in_area(output_label, "F2", "F3", scale_factor=0.7)
        self.place_in_area(output_line, "F4", "F6", scale_factor=0.9)

        # 4. Dots
        num_dots = 12
        x_values = np.linspace(0, PI/2, num_dots)
        input_dots = VGroup(*[Dot(input_line.n2p(x), color="#87CEEB", radius=0.06) for x in x_values])
        output_dots = VGroup(*[Dot(output_line.n2p(np.sin(x)), color="#FFD700", radius=0.06) for x in x_values])

        # === Animation for Lecture Line 1 ===
        # "Equally spaced input dots represent a uniform start."
        self.lecture[0].set_color("#FFFF00")
        self.play(Create(input_line), FadeIn(input_label))
        self.play(Create(input_dots, lag_ratio=0.1))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "Sine spreads dots at zero and bunches at pi-half."
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(output_line), FadeIn(output_label))
        self.play(Create(axes), Create(sine_graph), FadeIn(graph_label))
        
        # Transform input dots to output dots via f(x)=sin(x)
        self.play(
            ReplacementTransform(input_dots.copy(), output_dots),
            run_time=3,
            rate_func=rate_functions.ease_in_out_quad
        )
        self.wait(0.5)

        # Highlights and Density Labels
        # Highlight Spread (High Derivative f'(0)=1)
        rect_spread = SurroundingRectangle(VGroup(output_dots[0], output_dots[1]), color="#FFFFFF", buff=0.1)
        label_dilated = Text("Dilated", font_size=20, color="#FFFFFF").next_to(rect_spread, UP, buff=0.1)
        
        # Highlight Bunched (Low Derivative f'(pi/2)=0)
        rect_bunched = SurroundingRectangle(VGroup(output_dots[-2], output_dots[-1]), color="#FF0000", buff=0.1)
        label_contracted = Text("Contracted", font_size=20, color="#FF0000").next_to(rect_bunched, UP, buff=0.1)

        self.play(Create(rect_spread), Write(label_dilated))
        self.play(Indicate(VGroup(output_dots[0], output_dots[1]))) # L004: Correct 'Indicate'
        self.wait(1)
        
        self.play(Create(rect_bunched), Write(label_contracted))
        self.play(Indicate(VGroup(output_dots[-2], output_dots[-1]), color="#FF0000"))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "The derivative measures the resulting density of the mapping."
        self.lecture[2].set_color("#FFFF00")
        self.play(Write(formula))
        self.play(Write(density_text))
        
        # Visualize gaps (density ratio)
        # Using max_tip_length_to_length_ratio (L020) for small arrows
        arrow_in = DoubleArrow(
            input_dots[0].get_center(), input_dots[1].get_center(), 
            buff=0, color="#87CEEB", stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )
        arrow_out = DoubleArrow(
            output_dots[0].get_center(), output_dots[1].get_center(), 
            buff=0, color="#FFD700", stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )
        
        self.play(FadeIn(arrow_in), FadeIn(arrow_out))
        self.play(Indicate(formula))
        self.wait(2)
        
        # Cleanup
        self.play(
            FadeOut(rect_spread), FadeOut(rect_bunched), 
            FadeOut(label_dilated), FadeOut(label_contracted), 
            FadeOut(arrow_in), FadeOut(arrow_out)
        )
        self.wait(2)
