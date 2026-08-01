from manim import *
import numpy as np

class Section5Scene(Scene):
    def construct(self):
        # Define colors
        color_axes = WHITE
        color_ode = GREEN
        color_sol = RED
        color_curve = YELLOW

        # Configure Axes with a robust configuration for labels
        # Using label_constructor=Text ensures no LaTeX dependency is required for coordinates
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 20],
            x_length=8,
            y_length=5,
            axis_config={
                "include_tip": True, 
                "color": color_axes,
                "stroke_width": 2,
                "label_constructor": Text
            },
            tips=True
        ).add_coordinates()

        # Labels for the axes using Text
        x_axis_label = Text("t (time)", font_size=20).next_to(axes.x_axis.get_end(), DOWN)
        y_axis_label = Text("P (population)", font_size=20).next_to(axes.y_axis.get_end(), LEFT).rotate(90 * DEGREES).shift(UP * 0.5)

        # Equations using Text to ensure maximum compatibility across environments
        ode_equation = Text("dP/dt = kP", color=color_ode, font_size=32)
        ode_equation.to_corner(UL, buff=0.5)

        solution_equation = Text("P(t) = P0 * e^(kt)", color=color_sol, font_size=28)
        solution_equation.next_to(ode_equation, DOWN, buff=0.3, aligned_edge=LEFT)

        # Growth model constants
        p_zero = 10
        growth_rate = 0.6

        # Growth curve definition: P(t) = 10 * e^(0.6t)
        growth_curve = axes.plot(
            lambda t: p_zero * np.exp(growth_rate * t),
            x_range=[0, 3.8],
            color=color_curve,
            stroke_width=4
        )

        curve_label = Text("Exponential Growth", font_size=20, color=color_curve)
        curve_label.next_to(growth_curve.points[-1], UR, buff=0.2)

        # --- Animation Sequence ---
        
        # 1. Create coordinate system
        self.play(Create(axes), run_time=1.5)
        self.play(Write(x_axis_label), Write(y_axis_label))
        self.wait(0.5)
        
        # 2. Show the Differential Equation
        self.play(Write(ode_equation))
        self.wait(1)
        
        # 3. Show the Analytical Solution
        self.play(Write(solution_equation))
        self.wait(1)

        # 4. Animate the growth curve creation
        self.play(
            Create(growth_curve),
            Write(curve_label),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # 5. Highlight a specific point on the curve
        t_snapshot = 2.5
        p_snapshot = p_zero * np.exp(growth_rate * t_snapshot)
        point_coords = axes.c2p(t_snapshot, p_snapshot)
        
        target_dot = Dot(point_coords, color=WHITE, radius=0.08)
        projection_lines = axes.get_lines_to_point(point_coords)
        
        snapshot_label = Text(f"P({t_snapshot}) ≈ {int(p_snapshot)}", font_size=18)
        snapshot_label.next_to(target_dot, UL, buff=0.1)

        self.play(Create(projection_lines), FadeIn(target_dot, scale=0.5))
        self.play(Write(snapshot_label))
        
        self.wait(3)