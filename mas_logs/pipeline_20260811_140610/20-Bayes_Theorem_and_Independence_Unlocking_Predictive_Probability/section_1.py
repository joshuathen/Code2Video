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
        self.setup_layout("Prerequisite Warm-up: The Concept of Conditional Probability", [
            "Conditional probability focuses on a restricted sample space.",
            "Visualize all outcomes within the total set.",
            "New evidence acts as a filtering constraint."
        ])
        
        self.lecture[0].set_opacity(0)
        self.lecture[1].set_opacity(0)
        self.lecture[2].set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Fade in text 'Conditional Probability P(A|B)'
        self.lecture[0].set_color(WHITE)
        self.lecture[0].set_opacity(1)
        
        prob_text = Text("Conditional Probability P(A|B)", font_size=32).to_edge(UP, buff=0.8)
        self.play(Write(prob_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display Venn diagram showing overlapping sets A and B. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg]
        self.lecture[1].set_color(WHITE)
        self.lecture[1].set_opacity(1)
        
        circle_b = Circle(radius=1.2, color=BLUE, fill_opacity=0.3)
        circle_a = Circle(radius=1.2, color=YELLOW, fill_opacity=0.3).shift(LEFT * 0.8)
        
        venn = VGroup(circle_a, circle_b)
        lens_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        self.place_in_area(venn, 'A2', 'B4', scale_factor=0.65)
        self.place_at_grid(lens_icon, 'A5', scale_factor=0.5)
        
        intersection = Intersection(circle_a, circle_b, color="#FF5733", fill_opacity=0.8)
        
        self.play(FadeIn(venn), FadeIn(lens_icon))
        self.play(Create(intersection))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shrink reference space to set B only. Highlight ratio P(A|B) = P(A ∩ B) / P(B). [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/funnel.svg]
        self.lecture[2].set_color(WHITE)
        self.lecture[2].set_opacity(1)
        
        funnel_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/funnel.svg")
        self.place_at_grid(funnel_icon, 'D1', scale_factor=0.5)
        
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=36)
        self.place_at_grid(formula, 'D3', scale_factor=0.7)
        
        self.play(
            FadeOut(circle_a),
            FadeOut(lens_icon),
            FadeIn(funnel_icon),
            circle_b.animate.set_fill(opacity=0.5),
            ReplacementTransform(intersection, intersection.copy().scale(1.5).move_to(circle_b.get_center()))
        )
        self.play(Write(formula))
        self.wait(2)
