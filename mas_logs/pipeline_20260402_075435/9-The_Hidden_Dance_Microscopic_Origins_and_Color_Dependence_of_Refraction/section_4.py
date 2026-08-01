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
        # Setup the layout with the lecture content
        title_text = "The Math of Frequency Dependence"
        lecture_lines = [
            "Refractive index depends on the light's driving frequency.",
            "Higher frequencies closer to resonance feel stronger interactions.",
            "This causes n to increase for shorter wavelengths."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight line 1
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)

        # Create Axes for n vs Frequency (omega)
        # Using the area B1 to F6 for the graph
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[1, 4, 1],
            axis_config={"color": WHITE, "include_tip": True},
            x_length=4.5,
            y_length=4.0,
            tips=True
        )
        
        # Labels for the axes - Using Text instead of MathTex to avoid LaTeX dependency
        n_label = Text("n", slant=ITALIC, color=WHITE).scale(0.6)
        omega_label = Text("ω", slant=ITALIC, color=WHITE).scale(0.6)
        
        # Positioning axes in the right-side area
        axes_group = VGroup(axes, n_label, omega_label)
        self.place_in_area(axes_group, "B1", "F6")
        
        # Adjusting labels specifically near the axis tips
        n_label.next_to(axes.y_axis, UP, buff=0.1)
        omega_label.next_to(axes.x_axis, RIGHT, buff=0.1)

        self.play(Create(axes), Write(n_label), Write(omega_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2
        self.play(self.lecture[1].animate.set_color("#FFFF00"), run_time=0.5)

        # Resonance line at omega_0
        omega_0 = 4.5
        resonance_line = DashedLine(
            axes.c2p(omega_0, 1),
            axes.c2p(omega_0, 4),
            color="#FF0000"
        )
        # Using Unicode for omega and subscript 0
        resonance_label = Text("ω₀", color="#FF0000", slant=ITALIC).scale(0.5)
        resonance_label.next_to(resonance_line, DOWN, buff=0.1)

        # The curve: index of refraction increases as it approaches resonance
        curve = axes.plot(
            lambda x: 1 + 0.6 / (4.8 - x),
            x_range=[0, 4.3],
            color=WHITE
        )

        self.play(Create(resonance_line), Write(resonance_label))
        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)

        # Red light (Low frequency, far from resonance)
        red_w = 0.8
        red_n = curve.underlying_function(red_w)
        red_dot = Dot(axes.c2p(red_w, red_n), color="#FF0000")
        red_text = Text("Red", color="#FF0000", font_size=18).next_to(red_dot, UR, buff=0.1)

        # Blue light (High frequency, closer to resonance)
        blue_w = 3.8
        blue_n = curve.underlying_function(blue_w)
        blue_dot = Dot(axes.c2p(blue_w, blue_n), color="#0000FF")
        blue_text = Text("Blue", color="#0000FF", font_size=18).next_to(blue_dot, LEFT, buff=0.1)

        self.play(
            FadeIn(red_dot, scale=0.5),
            Write(red_text)
        )
        self.play(
            FadeIn(blue_dot, scale=0.5),
            Write(blue_text)
        )
        
        # Indicator showing n_blue > n_red - Using Text to avoid LaTeX
        # Addressing Issue 36: Move comparison to avoid cluttering axis label 'n'
        comparison = Text("n_blue > n_red", slant=ITALIC, color=WHITE)
        self.place_in_area(comparison, 'B4', 'C6', scale_factor=0.7)
        self.play(Write(comparison))
        
        self.wait(3)
