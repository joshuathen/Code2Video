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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Output: A Context-Rich Representation",
            [
                "- We multiply attention weights by the Value vectors.",
                "- This blends information from all relevant neighbors.",
                "- The result is a word enriched by its context."
            ]
        )

        # Define colors from storyboard
        value_color = "#00FF00"      # Value vector (#00FF00)
        bank_start_color = "#FFFFFF" # Chameleon icon starting color (#FFFFFF)
        bank_end_color = "#0000FF"   # Chameleon final blue color (#0000FF)

        # === Animation for Lecture Line 1 ===
        # "We multiply attention weights by the Value vectors."
        self.lecture[0].set_color(value_color)
        
        # Weight representation
        weight_val = MathTex("0.8", color=WHITE)
        self.place_at_grid(weight_val, "B3") # Fixed via Issue 35
        
        # Multiplication operator
        mult_op = MathTex("\\times", color=WHITE)
        self.place_at_grid(mult_op, "B4") # Fixed via Issue 35
        
        # Value vector representation
        value_rect = RoundedRectangle(height=0.6, width=1.6, color=value_color, fill_opacity=0.4)
        value_label = Text("Value (River)", font_size=18, color=WHITE)
        value_vector = VGroup(value_rect, value_label)
        self.place_at_grid(value_vector, "B5") # Fixed via Issue 35

        self.play(
            FadeIn(weight_val),
            FadeIn(mult_op),
            Create(value_vector)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "This blends information from all relevant neighbors."
        self.lecture[1].set_color(bank_end_color)
        
        # Chameleon representing the word 'Bank' (initially white/neutral)
        chameleon_body = Circle(radius=0.4, color=bank_start_color, fill_opacity=0.8)
        chameleon_label = Text("Bank", font_size=20, color=BLACK)
        chameleon = VGroup(chameleon_body, chameleon_label)
        self.place_at_grid(chameleon, "E4") # Fixed via Issue 36
        
        self.play(FadeIn(chameleon))
        self.wait(0.5)
        
        # Represent the blending/impact: a particle moves from the weighted Value to the Chameleon
        impact_dot = Dot(color=value_color).move_to(self.grid["B5"])
        
        self.play(
            impact_dot.animate.move_to(self.grid["E4"]),
            run_time=1.5,
            rate_func=bezier([0, 0, 1, 1])
        )
        
        # Color shift to blue upon blending
        self.play(
            chameleon_body.animate.set_color(bank_end_color),
            chameleon_label.animate.set_color(WHITE),
            FadeOut(impact_dot),
            Flash(self.grid["E4"], color=bank_end_color, line_length=0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The result is a word enriched by its context."
        self.lecture[2].set_color(bank_end_color)
        
        # Move the context-rich chameleon to the final output layer (bottom row)
        self.play(
            chameleon.animate.move_to(self.grid["F4"]),
            FadeOut(weight_val),
            FadeOut(mult_op),
            FadeOut(value_vector),
            run_time=1.5
        )
        
        # Glow/highlight effect to show enrichment
        glow = SurroundingRectangle(chameleon, color=bank_end_color, buff=0.1)
        self.play(Create(glow))
        self.play(
            Indicate(chameleon, color=bank_end_color, scale_factor=1.1),
            glow.animate.scale(1.2).set_opacity(0)
        )
        
        self.wait(2)
