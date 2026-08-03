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
        lecture_lines = [
            "Coefficients bn determine how much of each sine exists.",
            "We find them by multiplying the signal by sines.",
            "Large overlap area means the sine is a strong ingredient.",
            "If areas cancel out, that harmonic is not present.",
            "This projection extracts the specific recipe for our wave."
        ]
        self.setup_layout("The Projection Trick: Calculating Coefficients", lecture_lines)

        # Colors
        COLOR_SQUARE = "#FF0000"
        COLOR_SINE = "#00FF00"
        COLOR_AREA = "#FFFF00"

        # Assets
        tool_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tool.svg").set_color(WHITE)
        recipe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/recipe.svg").set_color(WHITE)

        # Pre-place axes
        axes = Axes(
            x_range=[0, 2 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False}
        )
        self.place_in_area(axes, "B1", "E6")

        # === Animation for Lecture Line 1 ===
        # "Coefficients bn determine how much of each sine exists."
        self.lecture[0].set_color(COLOR_SQUARE)
        
        square_wave = axes.plot(lambda x: 1 if np.sin(x) > 0 else -1, color=COLOR_SQUARE, use_smoothing=False)
        
        self.play(Write(axes))
        self.play(Create(square_wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We find them by multiplying the signal by sines."
        self.lecture[1].set_color(COLOR_SINE)
        
        # Place tool icon to indicate "operating"
        self.place_at_grid(tool_icon, "A6", scale_factor=0.5)
        sine_wave_n1 = axes.plot(lambda x: np.sin(x), color=COLOR_SINE)
        
        self.play(FadeIn(tool_icon, shift=DOWN))
        self.play(Create(sine_wave_n1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Large overlap area means the sine is a strong ingredient."
        self.lecture[2].set_color(COLOR_AREA)
        
        # For n=1, product is |sin(x)|
        product_wave_n1 = axes.plot(lambda x: abs(np.sin(x)), color=COLOR_AREA)
        area_fill_n1 = axes.get_area(product_wave_n1, x_range=[0, 2*PI], color=COLOR_AREA, opacity=0.3)
        
        label_b1 = MathTex("b_1 = \\text{Large Value}", color=COLOR_AREA, font_size=32)
        # Fix Issue #35: Center label above axes
        self.place_in_area(label_b1, "A1", "A6")
        
        # Place recipe icon next to label
        self.place_at_grid(recipe_icon, "A5", scale_factor=0.5)

        self.play(FadeOut(square_wave), FadeOut(sine_wave_n1), FadeOut(tool_icon))
        self.play(Create(product_wave_n1), FadeIn(area_fill_n1), Write(label_b1), FadeIn(recipe_icon))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # "If areas cancel out, that harmonic is not present."
        self.lecture[3].set_color(COLOR_AREA)
        
        # Return to Square and show Sine n=2
        square_wave_again = axes.plot(lambda x: 1 if np.sin(x) > 0 else -1, color=COLOR_SQUARE, use_smoothing=False)
        sine_wave_n2 = axes.plot(lambda x: np.sin(2 * x), color=COLOR_SINE)
        
        # Product for n=2: square(x) * sin(2x)
        product_wave_n2 = axes.plot(lambda x: (1 if np.sin(x) > 0 else -1) * np.sin(2 * x), color=COLOR_AREA)
        area_fill_n2 = axes.get_area(product_wave_n2, x_range=[0, 2*PI], color=COLOR_AREA, opacity=0.3)

        label_b2 = MathTex("b_2 = 0", color=COLOR_AREA, font_size=32)
        # Fix Issue #35: Center label above axes
        self.place_in_area(label_b2, "A1", "A6")

        self.play(
            FadeOut(product_wave_n1), 
            FadeOut(area_fill_n1), 
            FadeOut(label_b1),
            FadeOut(recipe_icon)
        )
        self.play(Create(square_wave_again), Create(sine_wave_n2))
        self.wait(1)
        self.play(
            FadeOut(square_wave_again),
            FadeOut(sine_wave_n2),
            Create(product_wave_n2),
            FadeIn(area_fill_n2),
            Write(label_b2)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "This projection extracts the specific recipe for our wave."
        self.lecture[4].set_color(WHITE)
        
        # Final Summary
        final_formula = MathTex("b_n = \\frac{1}{\\pi} \\int_0^{2\\pi} f(t) \\sin(nt) dt", color=WHITE, font_size=36)
        # Fix Issue #36: Center formula in visual area
        self.place_in_area(final_formula, "B2", "E5")
        
        # Re-introduce recipe icon near final formula
        self.place_at_grid(recipe_icon, "B6", scale_factor=0.6)
        
        self.play(FadeOut(product_wave_n2), FadeOut(area_fill_n2), FadeOut(label_b2))
        self.play(Write(final_formula), FadeIn(recipe_icon))
        self.wait(2)
