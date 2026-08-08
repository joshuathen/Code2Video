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
        title = "The Fundamental Theorem: Two Sides of a Coin"
        lines = [
            "The area's derivative is the original function's height.",
            "Differentiation and integration are inverse operations.",
            "Integrating a rate gives you the total accumulation.",
            "Differentiating accumulation returns the original rate.",
            "They are two sides of the same coin."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # L1: "The area's derivative is the original function's height."
        # Display f(x) (#00FF00) and its accumulation A(x) (#FF8C00)
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        fx_label = MathTex("f(x)", color="#00FF00")
        ax_label = MathTex("A(x)", color="#FF8C00")
        
        self.place_in_area(fx_label, 'B3', 'B4', scale_factor=1.2)
        self.place_in_area(ax_label, 'E3', 'E4', scale_factor=1.2)
        
        self.play(Write(fx_label), Write(ax_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: "Differentiation and integration are inverse operations."
        self.play(self.lecture[1].animate.set_color("#FF8C00"))
        # Briefly highlight the relationship by pulsing both
        self.play(
            fx_label.animate(rate_func=there_and_back).scale(1.1),
            ax_label.animate(rate_func=there_and_back).scale(1.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L3: "Integrating a rate gives you the total accumulation."
        # Draw a cyan arrow labeled "Integrate" (#00FFFF) from f(x) to A(x).
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        integrate_arrow = CurvedArrow(
            start_point=self.grid['B5'] + RIGHT*0.2, 
            end_point=self.grid['E5'] + RIGHT*0.2, 
            angle=-PI/1.2, 
            color="#00FFFF"
        )
        integrate_text = Text("Integrate", font_size=18, color="#00FFFF")
        self.place_at_grid(integrate_text, 'C5') # Fixed: Issue 39 (moved from C6)
        
        # Pulse A(x) while the "Integrate" arrow is active.
        self.play(
            Create(integrate_arrow), 
            Write(integrate_text),
            ax_label.animate(rate_func=there_and_back).scale(1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # L4: "Differentiating accumulation returns the original rate."
        # Draw a magenta arrow labeled "Differentiate" (#FF00FF) from A(x) back to f(x).
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        
        diff_arrow = CurvedArrow(
            start_point=self.grid['E2'] + LEFT*0.2, 
            end_point=self.grid['B2'] + LEFT*0.2, 
            angle=-PI/1.2, 
            color="#FF00FF"
        )
        diff_text = Text("Differentiate", font_size=18, color="#FF00FF")
        self.place_at_grid(diff_text, 'D2') # Fixed: Issue 38 (moved from D1)
        
        self.play(
            Create(diff_arrow), 
            Write(diff_text),
            fx_label.animate(rate_func=there_and_back).scale(1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # L5: "They are two sides of the same coin."
        # Show the text "Fundamental Theorem of Calculus" (#FFFFFF) centered between the functions.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        ftc_text = Text("Fundamental Theorem\nof Calculus", font_size=22, color=WHITE, line_spacing=0.8)
        self.place_in_area(ftc_text, 'C3', 'D4')
        
        # Visual loop emphasis
        loop_group = VGroup(fx_label, ax_label, integrate_arrow, diff_arrow, integrate_text, diff_text)
        self.play(
            Flash(loop_group, color=WHITE, line_length=0.4, flash_radius=2.0),
            Write(ftc_text)
        )
        self.wait(2)
