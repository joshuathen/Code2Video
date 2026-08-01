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

class Section2Scene(TeachingScene):
    def construct(self):
        config.tex_compiler = "pdflatex"
        # Setup the layout with the title and lecture lines
        self.setup_layout(
            "Prerequisite Check: The Chain Rule Engine",
            [
                "The Chain Rule is our engine for implicit differentiation.",
                "Treat y as a nested function inside x.",
                "Differentiating y cubed gives three y squared times dy/dx."
            ]
        )

        # Define Colors
        WHITE_COL = "#FFFFFF"
        YELLOW_COL = "#F7D038"
        GREEN_COL = "#009E73"
        GRAY_COL = "#888888"

        # === Animation for Lecture Line 1 ===
        # Stage 1: Display 'd/dx [ y³ ]' in White (#FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE_COL))
        
        expr1 = MathTex(r"\frac{d}{dx}", r"[", r"y", r"^3", r"]", color=WHITE_COL)
        self.place_in_area(expr1, "B2", "C5", scale_factor=1.5)
        self.play(Write(expr1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stage 2: Transform 'y' into 'y(x)' in Yellow (#F7D038)
        self.play(
            self.lecture[0].animate.set_color(GRAY_COL),
            self.lecture[1].animate.set_color(YELLOW_COL)
        )
        
        # Creating a version with y(x) explicitly as a function
        expr2 = MathTex(r"\frac{d}{dx}", r"[", r"(", r"y(x)", r")", r"^3", r"]", color=WHITE_COL)
        expr2.set_color_by_tex("y(x)", YELLOW_COL)
        self.place_in_area(expr2, "B2", "C5", scale_factor=1.5)
        
        # Smooth transformation of the expression to show y as y(x)
        self.play(ReplacementTransform(expr1, expr2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stage 3: Show the Power Rule step: '3(y(x))²'
        self.play(
            self.lecture[1].animate.set_color(GRAY_COL),
            self.lecture[2].animate.set_color(GREEN_COL)
        )
        
        expr3 = MathTex(r"3", r"(", r"y(x)", r")", r"^2", color=WHITE_COL)
        expr3.set_color_by_tex("y(x)", YELLOW_COL)
        self.place_at_grid(expr3, "C3", scale_factor=1.5)
        
        # Animate the transition to the outer derivative result
        self.play(
            FadeOut(expr2[0], shift=UP),    # remove d/dx
            FadeOut(expr2[1], shift=LEFT),  # remove [
            FadeOut(expr2[6], shift=RIGHT), # remove ]
            ReplacementTransform(expr2[2:6], expr3)
        )
        self.wait(0.5)
        
        # Stage 4: Animate the 'tail' (dy/dx) in Green (#009E73) sliding in from the right.
        tail = MathTex(r"\cdot", r"\frac{dy}{dx}", color=GREEN_COL)
        self.place_at_grid(tail, "C4", scale_factor=1.5)
        
        self.play(FadeIn(tail, shift=LEFT))
        self.wait(1)

        # Stage 5: Group terms into the final result '3y² * dy/dx' and make it glow.
        final_expr = MathTex(r"3", r"y^2", r"\cdot", r"\frac{dy}{dx}")
        final_expr[0:2].set_color(WHITE_COL)
        final_expr[2:].set_color(GREEN_COL)
        self.place_in_area(final_expr, "D2", "E5", scale_factor=1.5)
        
        # Final condensation of the nested notation back to simpler y notation
        self.play(ReplacementTransform(VGroup(expr3, tail), final_expr))
        
        # Glow effect to emphasize the completed differentiation
        glow = final_expr.copy().set_stroke(width=10, opacity=0.4).set_color(GREEN_COL)
        self.play(FadeIn(glow))
        self.play(FadeOut(glow))
        self.wait(2)
