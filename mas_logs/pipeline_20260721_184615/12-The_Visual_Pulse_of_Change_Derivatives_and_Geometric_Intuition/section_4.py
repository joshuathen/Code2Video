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
        # Retrieve data from shared state
        title_text = "The Power Rule: Algebra Meets Geometry"
        lecture_lines = [
            "Meet the power rule, a shortcut for finding slopes.",
            "For any power function, multiply and subtract exponents.",
            "Consider a skateboard ramp shaped like x squared.",
            "This formula calculates the slope at any point.",
            "At position three, the ramp's steepness is exactly six."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color("#00FFFF")
        
        # Display the Power Rule formula 'd/dx [x^n] = nx^(n-1)' in cyan (#00FFFF) with a surrounding rectangle.
        formula = MathTex(
            r"\frac{d}{dx}", r"[x^n]", "=", "n", r"x^{n-1}", 
            color="#00FFFF"
        )
        rect = SurroundingRectangle(formula, color="#FFFFFF", buff=0.2)
        formula_group = VGroup(formula, rect)
        
        # Issue 29: Position formula_group at C2-D5 to balance vertical layout
        self.place_in_area(formula_group, "C2", "D5", scale_factor=1.0)
        
        self.play(Write(formula), Create(rect))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#00FFFF")
        
        # Highlight logic: "multiply and subtract exponents"
        # Using Indicate (L004) to highlight formula parts
        self.play(Indicate(formula[1]), Indicate(formula[3], color="#00FFFF"))
        self.wait(1.0)
        self.play(Indicate(formula[4], color="#00FFFF"))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#00FFFF")
        
        # Transition: Clear formula to make room for geometric example
        self.play(FadeOut(formula_group))
        
        # Issue 30: Axes at C1-F6 for better headroom
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 11, 2],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        self.place_in_area(axes, "C1", "F6", scale_factor=0.8)
        
        parabola = axes.plot(lambda x: x**2, x_range=[-0.5, 3.3], color="#FFFFFF")
        parabola_label = MathTex("y = x^2", color="#FFFFFF", font_size=24)
        # Position label using c2p within 1 grid unit (L002)
        parabola_label.next_to(axes.c2p(2.5, 6.25), LEFT, buff=0.2)
        
        # Issue 20: Asset integration (ramp.svg)
        # Load ramp icon to visually model the parabola
        ramp_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ramp.svg")
        self.place_in_area(ramp_asset, "C1", "F6", scale_factor=0.8)
        # Apply opacity for background feel (L031)
        ramp_asset.set_stroke(opacity=0.3).set_fill(opacity=0.1)
        
        self.play(Create(axes), Create(parabola), Write(parabola_label), FadeIn(ramp_asset))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color("#00FFFF")
        
        # x=2 initially as per storyboard
        x_val = ValueTracker(2)
        
        # Issue 20: Asset integration (skateboard.svg)
        skateboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/skateboard.svg")
        skateboard.scale(0.3)
        # Use updater for efficient movement (L027/MAS Instruction 10)
        skateboard.add_updater(lambda m: m.move_to(axes.c2p(x_val.get_value(), x_val.get_value()**2) + UP * 0.2))
        
        # Tangent line with updater
        tangent_line = Line(color="#00FFFF")
        def update_tangent(line):
            x = x_val.get_value()
            slope = 2 * x
            y = x**2
            line.set_points_by_ends(
                axes.c2p(x - 0.8, y - 0.8 * slope),
                axes.c2p(x + 0.8, y + 0.8 * slope)
            )
        tangent_line.add_updater(update_tangent)
        
        # Slope label using DecimalNumber for updates
        slope_label_text = Text("Slope = ", font_size=20, color="#00FFFF")
        slope_num = DecimalNumber(4, num_decimal_places=0, font_size=20, color="#00FFFF")
        slope_group = VGroup(slope_label_text, slope_num).arrange(RIGHT, buff=0.1)
        
        # Follow Proximity Rule (L002)
        slope_group.add_updater(lambda m: m.next_to(axes.c2p(x_val.get_value(), x_val.get_value()**2), DR, buff=0.2))
        slope_num.add_updater(lambda m: m.set_value(2 * x_val.get_value()))
        
        self.play(FadeIn(skateboard), Create(tangent_line), Write(slope_group))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color("#00FFFF")
        
        # Move to x=3 to demonstrate the calculation: d/dx(x^2) = 2x => 2(3) = 6
        # Aligned with Lecture Line 5: "At position three, the ramp's steepness is exactly six."
        self.play(x_val.animate.set_value(3), run_time=2.0, rate_func=rate_functions.ease_in_out_quad)
        self.wait(2.0)
