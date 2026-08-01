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

class Section5Scene(TeachingScene):
    def construct(self):
        title_text = "The Big Three: Classification of PDEs"
        lecture_lines = [
            "PDEs fall into three main types based on behavior.",
            "Elliptic equations describe steady-state systems and equilibrium.",
            "Parabolic equations model diffusion and spreading over time.",
            "Hyperbolic equations represent waves and transport phenomena.",
            "This classification helps us choose the right solving strategy."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # PDEs fall into three main types based on behavior.
        # Display 'PDE Classification' in white (#FFFFFF) at the top.
        self.lecture[0].set_color(WHITE)
        pde_class_label = Text("PDE Classification", font_size=32, color=WHITE)
        self.place_in_area(pde_class_label, "A2", "A5")
        self.play(Write(pde_class_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Elliptic equations describe steady-state systems and equilibrium.
        # Show 'Elliptic: Equilibrium' (#00FF00) next to a static smooth surface.
        self.lecture[1].set_color("#00FF00")
        elliptic_label = Text("Elliptic: Equilibrium", font_size=20, color="#00FF00")
        self.place_at_grid(elliptic_label, "B3", scale_factor=0.8)
        
        elliptic_surface = Line(LEFT, RIGHT, color="#00FF00")
        self.place_at_grid(elliptic_surface, "B4", scale_factor=1.0)
        
        self.play(FadeIn(elliptic_label), Create(elliptic_surface))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Parabolic equations model diffusion and spreading over time.
        # Show 'Parabolic: Diffusion' (#FFFF00) with a spreading Gaussian curve.
        self.lecture[2].set_color("#FFFF00")
        parabolic_label = Text("Parabolic: Diffusion", font_size=20, color="#FFFF00")
        self.place_at_grid(parabolic_label, "C3", scale_factor=0.8)
        
        sigma_tracker = ValueTracker(0.2)
        curve_pos = self.grid["C4"]
        
        # Helper to get Gaussian points
        def get_gaussian_points(sigma):
            return ParametricFunction(
                lambda t: np.array([t, 0.4 * (1/sigma) * np.exp(-t**2 / (2 * sigma**2)), 0]),
                t_range=[-1.5, 1.5]
            ).scale(0.6).move_to(curve_pos).get_points()

        parabolic_curve = ParametricFunction(
            lambda t: np.array([t, 0.4 * (1/0.2) * np.exp(-t**2 / (2 * 0.2**2)), 0]),
            t_range=[-1.5, 1.5],
            color="#FFFF00"
        ).scale(0.6).move_to(curve_pos)
        
        # Update curve points in-place to optimize performance
        parabolic_curve.add_updater(lambda m: m.set_points(get_gaussian_points(sigma_tracker.get_value())))
        
        self.play(FadeIn(parabolic_label), Create(parabolic_curve))
        self.play(sigma_tracker.animate.set_value(1.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Hyperbolic equations represent waves and transport phenomena.
        # Show 'Hyperbolic: Waves' (#00FFFF) with two pulses moving in opposite directions.
        self.lecture[3].set_color("#00FFFF")
        hyperbolic_label = Text("Hyperbolic: Waves", font_size=20, color="#00FFFF")
        self.place_at_grid(hyperbolic_label, "D3", scale_factor=0.8)
        
        pulse_func = lambda x: 0.4 * np.exp(-x**2 / 0.05)
        pulse1 = ParametricFunction(lambda t: [t, pulse_func(t), 0], t_range=[-0.5, 0.5], color="#00FFFF")
        pulse2 = ParametricFunction(lambda t: [t, pulse_func(t), 0], t_range=[-0.5, 0.5], color="#00FFFF")
        
        self.place_at_grid(pulse1, "D4", scale_factor=0.8)
        self.place_at_grid(pulse2, "D4", scale_factor=0.8)
        
        self.play(FadeIn(hyperbolic_label), Create(pulse1), Create(pulse2))
        self.play(
            pulse1.animate.shift(LEFT * 1.2),
            pulse2.animate.shift(RIGHT * 1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This classification helps us choose the right solving strategy.
        # Display 'Fundamental Physics' (#FFFFFF) below the three examples.
        self.lecture[4].set_color(WHITE)
        fundamental_physics = Text("Fundamental Physics", font_size=28, color=WHITE)
        self.place_in_area(fundamental_physics, "E2", "E5")
        
        self.play(Write(fundamental_physics))
        self.wait(2)
