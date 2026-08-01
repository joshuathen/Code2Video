from manim import *

class Section4Scene(Scene):
    def construct(self):
        # 1. Setup Coordinate System
        axes = Axes(
            x_range=[-7, 7, 1],
            y_range=[-7, 7, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True}
        ).to_edge(RIGHT, buff=0.5)
        
        # 2. Create the Circle (x^2 + y^2 = 25)
        # Using radius=5 in the axes coordinate space
        circle = axes.plot_implicit_curve(
            lambda x, y: x**2 + y**2 - 25,
            color=BLUE
        )
        
        # 3. Define Labels and Points
        # Replaced MathTex with Text due to missing LaTeX environment
        eq_title = Text("x² + y² = 25", color=BLUE).to_corner(UL)
        
        # Target point (3, 4)
        dot_coord = [3, 4, 0]
        dot = Dot(axes.c2p(3, 4), color=RED)
        dot_label = Text("(3, 4)", font_size=24).next_to(dot, UR, buff=0.1)
        
        # 4. Derivation Steps (Implicit Differentiation)
        # Replaced MathTex with Text to avoid LaTeX dependency
        steps = VGroup(
            Text("d/dx(x² + y²) = d/dx(25)"),
            Text("2x + 2y (dy/dx) = 0"),
            Text("2y (dy/dx) = -2x"),
            Text("dy/dx = -x/y")
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).next_to(eq_title, DOWN, buff=0.5, aligned_edge=LEFT)
        
        # 5. Calculate Slope and Tangent Line
        # At (3, 4), slope m = -3/4 = -0.75
        slope_calc = Text("m = -3/4", color=YELLOW).scale(0.8).next_to(steps, DOWN, buff=0.5, aligned_edge=LEFT)
        
        # Tangent line equation: y - 4 = -0.75(x - 3) => y = -0.75x + 6.25
        tangent_line = axes.plot(
            lambda x: -0.75 * x + 6.25,
            x_range=[-1, 7],
            color=YELLOW
        )

        # 6. Animation Sequence
        self.play(Create(axes), run_time=1)
        self.play(Create(circle), Write(eq_title), run_time=1.5)
        self.wait(0.5)
        
        self.play(FadeIn(dot, scale=0.5), Write(dot_label))
        self.wait(0.5)
        
        # Animate Differentiation Steps
        for step in steps:
            self.play(Write(step))
            self.wait(0.3)
            
        self.play(Write(slope_calc))
        self.play(Create(tangent_line))
        self.wait(2)