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
        # Section Title and Lecture Lines
        title_text = "Operation 2: Finding the Root"
        lecture_lines = [
            "To find a root, look at the bottom left.",
            "We seek the foundation for this exponent and result.",
            "The cube root of twenty-seven reveals the base three."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors for the Triangle of Power components
        COLOR_EXP = "#F1C40F"  # Yellow for Exponent
        COLOR_RES = "#E67E22"  # Orange for Result
        COLOR_BASE = "#3498DB" # Blue for Base (Root)
        
        # Positioning based on Beliefs:
        # B021: Min gap from lecture. Using Cols 2-6.
        # B005: Avoid Row A and Row F for primary labels to stay clear of title/edges.
        # We will use Rows B, C, D, E.
        
        # Triangle Vertices (Points)
        v_top = self.grid["C4"]
        v_left = self.grid["E3"]
        v_right = self.grid["E5"]
        
        triangle = Polygon(v_top, v_right, v_left, color=WHITE, stroke_width=3)
        
        # Labels (B012: one unit away from vertices)
        # Exponent at B4 (exactly one unit above C4)
        exponent = MathTex("3", color=COLOR_EXP)
        self.place_at_grid(exponent, "B4", scale_factor=1.4)
        
        # Result at E6 (exactly one unit right of E5)
        result = MathTex("27", color=COLOR_RES)
        self.place_at_grid(result, "E6", scale_factor=1.4)
        
        # Base "?" at E2 (exactly one unit left of E3)
        base_q = MathTex("?", color=COLOR_BASE)
        self.place_at_grid(base_q, "E2", scale_factor=1.6)
        
        # Final Base "3" at the same position (E2)
        base_val = MathTex("3", color=COLOR_BASE)
        self.place_at_grid(base_val, "E2", scale_factor=1.4)

        # === Animation for Lecture Line 1 ===
        # "To find a root, look at the bottom left."
        self.play(self.lecture[0].animate.set_color(COLOR_BASE))
        self.play(Create(triangle))
        self.play(FadeIn(exponent), FadeIn(result))
        
        # Animation Description: "In the Triangle, bold the Exponent and Result values."
        self.play(
            exponent.animate.set_stroke(width=2),
            result.animate.set_stroke(width=2),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We seek the foundation for this exponent and result."
        self.play(self.lecture[1].animate.set_color(COLOR_BASE))
        
        # Animation Description: "Show a flashing '?' at the Base vertex."
        # Flashing effect at E2
        for _ in range(2):
            self.play(FadeIn(base_q), run_time=0.3)
            self.play(FadeOut(base_q), run_time=0.3)
        self.play(FadeIn(base_q))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The cube root of twenty-seven reveals the base three."
        self.play(self.lecture[2].animate.set_color(COLOR_BASE))
        
        # Animation Description: "Transform the '?' into the number '3'" 
        # (Corrected storyboard '2' to '3' to match lecture text and math: cuberoot(27)=3)
        # Also return Exponent and Result to normal weight.
        self.play(
            ReplacementTransform(base_q, base_val),
            exponent.animate.set_stroke(width=0),
            result.animate.set_stroke(width=0),
            run_time=1
        )
        self.wait(2)
