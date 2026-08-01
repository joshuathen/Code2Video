from manim import *
import numpy as np

class Section3Scene(Scene):
    def construct(self):
        # 1. Setup Title
        title = Text("Implicit Differentiation", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.5)
        
        # 2. Setup Lecture Content (Left Side)
        # Replaced MathTex with Text to avoid LaTeX dependency (fixes FileNotFoundError: 'latex')
        lecture_items = [
            "• Variables x and y are linked",
            "• y is a function of x: y(x)",
            "• Use the Chain Rule",
            "• Differentiate both sides"
        ]
        
        lecture_group = VGroup(*[
            Text(item, font_size=24, color=WHITE) for item in lecture_items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        lecture_group.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        
        # 3. Setup Workspace (Right Side)
        workspace_rect = RoundedRectangle(
            height=5, width=6, corner_radius=0.2, color=BLUE_B, fill_opacity=0.1
        ).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)
        
        workspace_center = workspace_rect.get_center()
        
        # 4. Mathematical Content
        # Using Text with unicode characters to bypass LaTeX requirement while maintaining readability
        eq1 = Text("x² + y² = 25", font_size=34).move_to(workspace_center + UP * 1.5)
        eq2 = Text("d/dx(x²) + d/dx(y²) = d/dx(25)", font_size=22).move_to(workspace_center + UP * 0.5)
        eq3 = Text("2x + 2y · dy/dx = 0", font_size=34).move_to(workspace_center + DOWN * 0.5)
        eq4 = Text("dy/dx = -x/y", font_size=34, color=YELLOW).move_to(workspace_center + DOWN * 1.5)

        # 5. Animations
        self.play(Write(title))
        self.wait(0.5)
        
        self.play(Create(lecture_group), run_time=2)
        self.wait(0.5)
        
        self.play(Create(workspace_rect))
        self.play(Write(eq1))
        self.wait(1)
        
        # TransformMatchingShapes works with Text submobjects (characters)
        self.play(TransformMatchingShapes(eq1.copy(), eq2))
        self.wait(1)
        
        self.play(Write(eq3))
        self.wait(1)
        
        self.play(FadeIn(eq4, shift=UP))
        self.play(Indicate(eq4))
        
        self.wait(2)

    def get_grid_pos(self, row, col):
        """
        Helper to map a 4x4 workspace grid on the right side.
        Rows: 0 (top) to 3 (bottom)
        Cols: 0 (left) to 3 (right)
        """
        start_x = 1.5
        start_y = 1.5
        x_step = 1.5
        y_step = 1.0
        return np.array([start_x + col * x_step, start_y - row * y_step, 0])