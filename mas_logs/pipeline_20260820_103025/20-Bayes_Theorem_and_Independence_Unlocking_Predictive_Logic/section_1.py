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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Conditional probability measures A given B occurred.",
            "It restricts the sample space to event B.",
            "Example: Knowing a card is Red changes probabilities."
        ]
        self.setup_layout("Prerequisite Review: Conditional Probability", lecture_lines)
        
        # Assets
        card_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png")
        
        # Create Circles for Venn Diagram
        circle_a = Circle(color="#FF5733", fill_opacity=0.4)
        circle_b = Circle(color="#33FF57", fill_opacity=0.4)
        
        # Positioning Venn Diagram
        self.place_in_area(circle_a, 'A3', 'C4', scale_factor=0.8)
        self.place_in_area(circle_b, 'A4', 'C5', scale_factor=0.8)
        
        # Intersection
        intersection = Intersection(circle_a, circle_b, color="#FFFF33", fill_opacity=0.6)
        
        # Formula
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=36)
        self.place_in_area(formula, 'D2', 'E5', scale_factor=1.0)
        
        # Asset Placement
        self.place_at_grid(card_icon, 'A2', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(FadeIn(circle_a), FadeIn(circle_b), FadeIn(card_icon))
        self.play(FadeIn(intersection))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        # Highlight B as new sample space
        highlight_b = Circle(color="#33FF57", fill_opacity=0.2, stroke_width=4)
        self.place_at_grid(highlight_b, 'B4', scale_factor=0.7)
        self.play(FadeIn(highlight_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        # Final Summary Asset
        final_summary = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png")
        self.place_at_grid(final_summary, 'F5', scale_factor=0.4)
        
        # Fade out original sample space, focus on intersection/B
        self.play(
            FadeOut(circle_a),
            FadeOut(circle_b),
            FadeOut(highlight_b),
            FadeIn(final_summary),
            intersection.animate.set_fill(opacity=1),
            formula.animate.set_color(YELLOW)
        )
        self.wait(2)
