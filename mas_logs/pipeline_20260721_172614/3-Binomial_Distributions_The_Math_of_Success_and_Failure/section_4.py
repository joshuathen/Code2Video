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
        title = "Visualizing the Formula"
        lines = [
            "This formula calculates the probability of specific outcomes.",
            "nCk counts the different ways successes can occur.",
            "p to the k is the probability of successes.",
            "q to the n minus k covers the failures.",
            "Combine these building blocks to find the total probability."
        ]
        self.setup_layout(title, lines)

        # Define colors
        COLOR_NCK = "#FFFF00"  # Yellow
        COLOR_PK = "#00FF00"   # Green
        COLOR_QK = "#FF0000"   # Red

        # === Animation for Lecture Line 1 ===
        # Display the formula P(X=k) = nCk * p^k * q^(n-k) in large white font.
        self.lecture[0].set_color(WHITE)
        
        # Using MathTex for the formula parts to allow highlighting
        formula = MathTex(
            "P(X=k)", "=", "{n \choose k}", "p^k", "q^{n-k}",
            font_size=40
        )
        self.place_in_area(formula, "A1", "B6")
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight 'nCk' in yellow (#FFFF00) and show 5 slots rearranging 3 successes.
        self.play(
            self.lecture[0].animate.set_color(GRAY), 
            self.lecture[1].animate.set_color(COLOR_NCK),
            formula[2].animate.set_color(COLOR_NCK)
        )
        
        # 5 slots rearranging 3 successes (S S S F F)
        slots = VGroup(*[Square(side_length=0.4, color=WHITE) for _ in range(5)]).arrange(RIGHT, buff=0.1)
        # Fix: Issue 31: Use 'C1' to 'C6'
        self.place_in_area(slots, "C1", "C6")
        
        s_texts = VGroup(*[Text("S", color=COLOR_NCK, font_size=20) for _ in range(3)])
        f_texts = VGroup(*[Text("F", color=WHITE, font_size=20) for _ in range(2)])
        tokens = VGroup(*s_texts, *f_texts)
        
        # Position tokens inside slots initially
        for i, token in enumerate(tokens):
            token.move_to(slots[i].get_center())
            
        self.play(Create(slots), FadeIn(tokens))
        self.wait(0.5)
        
        # Animation: Rearrange once to show "different ways"
        # Move S to 1,3,5 and F to 2,4 (indices 0, 2, 4 and 1, 3)
        new_targets = [slots[0].get_center(), slots[2].get_center(), slots[4].get_center(), 
                       slots[1].get_center(), slots[3].get_center()]
        self.play(*[tokens[i].animate.move_to(new_targets[i]) for i in range(5)])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight 'p^k' in green (#00FF00) and show the text 'p * p * p'.
        self.play(
            self.lecture[1].animate.set_color(GRAY), 
            self.lecture[2].animate.set_color(COLOR_PK),
            formula[3].animate.set_color(COLOR_PK)
        )
        
        p_expansion = MathTex("p \cdot p \cdot p", color=COLOR_PK, font_size=32)
        # Fix: Issue 32: Use 'D1' to 'D3'
        self.place_in_area(p_expansion, "D1", "D3")
        self.play(FadeIn(p_expansion))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight 'q^(n-k)' in red (#FF0000) and show the text 'q * q'.
        self.play(
            self.lecture[2].animate.set_color(GRAY), 
            self.lecture[3].animate.set_color(COLOR_QK),
            formula[4].animate.set_color(COLOR_QK)
        )
        
        q_expansion = MathTex("q \cdot q", color=COLOR_QK, font_size=32)
        # Fix: Issue 33: Use 'D4' to 'D6'
        self.place_in_area(q_expansion, "D4", "D6")
        self.play(FadeIn(q_expansion))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # All three highlighted formula parts slide together to form the complete equation.
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color(WHITE))
        
        # Clear the "visual aids"
        self.play(
            FadeOut(slots), 
            FadeOut(tokens), 
            FadeOut(p_expansion), 
            FadeOut(q_expansion)
        )
        
        # Final emphasis on the formula
        rect = SurroundingRectangle(formula[2:], color=WHITE, buff=0.1)
        self.play(Create(rect))
        self.wait(2)
        self.play(FadeOut(rect))
        self.wait(1)
