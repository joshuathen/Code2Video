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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "This connection is the Fundamental Theorem of Calculus.",
            "To find an integral, find its antiderivative F.",
            "The total area is the change in F.",
            "We calculate this by subtracting F(a) from F(b).",
            "Integration and differentiation are officially inverse processes."
        ]
        self.setup_layout("Formalizing the Relationship (FTC)", lecture_lines)
        
        # Colors
        WHITE_COLOR = "#FFFFFF"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#1E90FF"
        YELLOW_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "This connection is the Fundamental Theorem of Calculus."
        self.lecture[0].set_color(YELLOW_HIGHLIGHT)
        
        # Display the integral symbol and equation ∫ f(x) dx (#FFFFFF).
        # We define the full equation but will reveal it in stages.
        ftc_eq = MathTex(
            "\\int", "_{a}", "^{b}", "f(x)", "dx", "=", "F(b)", "-", "F(a)",
            font_size=42, color=WHITE_COLOR
        )
        # Fix for Issue 34: Change scale_factor to 1.0
        self.place_in_area(ftc_eq, 'B1', 'C6', scale_factor=1.0)
        
        left_side = ftc_eq[0:5]
        self.play(Write(left_side))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "To find an integral, find its antiderivative F."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_HIGHLIGHT)
        
        # Introduce the capital F(x) (#00FF00) as the 'Antiderivative'.
        equals_sign = ftc_eq[5]
        right_side = ftc_eq[6:9]
        
        # Color the F's green
        ftc_eq[6].set_color(GREEN_COLOR) # F(b)
        ftc_eq[8].set_color(GREEN_COLOR) # F(a)
        
        label_f = Text("Antiderivative", font_size=24, color=GREEN_COLOR)
        # Fix for Issue 33: Position label near the F(b) and F(a) parts at D5 with scale 0.8
        self.place_at_grid(label_f, 'D5', scale_factor=0.8)
        
        self.play(
            Write(equals_sign),
            Write(right_side),
            run_time=1.5
        )
        self.play(FadeIn(label_f, shift=UP*0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The total area is the change in F."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW_HIGHLIGHT)
        
        # Focus on the right side F(b) - F(a)
        self.play(Indicate(right_side, color=BLUE_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We calculate this by subtracting F(a) from F(b)."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW_HIGHLIGHT)
        
        # Show blue arrows (#1E90FF) pointing to boundaries.
        # Arrow from b in integral to F(b)
        # Arrow from a in integral to F(a)
        arrow_b = Arrow(
            start=ftc_eq[2].get_top() + UP*0.5, 
            end=ftc_eq[6].get_top(), 
            color=BLUE_COLOR, 
            buff=0.1,
            path_arc=-0.5
        )
        arrow_a = Arrow(
            start=ftc_eq[1].get_bottom() + DOWN*0.5, 
            end=ftc_eq[8].get_bottom(), 
            color=BLUE_COLOR, 
            buff=0.1,
            path_arc=0.5
        )

        self.play(
            Create(arrow_b),
            Create(arrow_a),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Integration and differentiation are officially inverse processes."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW_HIGHLIGHT)
        
        # Flash the entire equation to emphasize the completed connection.
        self.play(
            FadeOut(arrow_b),
            FadeOut(arrow_a),
            FadeOut(label_f)
        )
        self.play(Circumscribe(ftc_eq, color=WHITE_COLOR, fade_out=True, run_time=2))
        self.wait(1)
        
        # Reset color
        self.lecture[4].set_color(WHITE)
        self.wait(2)
