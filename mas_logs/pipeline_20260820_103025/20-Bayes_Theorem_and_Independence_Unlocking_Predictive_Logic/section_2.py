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
        self.setup_layout("Defining Independence", [
            "Independent events share no predictive information.", 
            "Mathematically: P(A|B) equals P(A).", 
            "Example: Coin flips are unaffected by cats."
        ])
        
        # Assets
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        # Check if the asset exists or provide a dummy shape/mobject if the file is missing to prevent OSError
        cat = Circle(radius=0.3, color=WHITE, fill_opacity=1) 

        # Elements
        circle_a = Circle(radius=0.5, color=BLUE, fill_opacity=0.3)
        label_a = MathTex("A").scale(0.7)
        label_a.next_to(circle_a, UP, buff=0.1)
        
        circle_b = Circle(radius=0.5, color=GREEN, fill_opacity=0.3)
        label_b = MathTex("B").scale(0.7)
        label_b.next_to(circle_b, UP, buff=0.1)
        
        formula = MathTex("P(A|B) = P(A)").scale(1.2)
        
        # Repositioning based on critical feedback
        self.place_at_grid(circle_a, "B4", scale_factor=1.0)
        self.place_at_grid(label_a, "B4", scale_factor=0.7)
        label_a.next_to(circle_a, UP, buff=0.1)

        self.place_at_grid(circle_b, "B5", scale_factor=1.0)
        self.place_at_grid(label_b, "B5", scale_factor=0.7)
        label_b.next_to(circle_b, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(circle_a), FadeIn(label_a), FadeIn(circle_b), FadeIn(label_b))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_in_area(formula, 'C4', 'D5', scale_factor=1.2)
        self.play(Write(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Flash circles with coin and cat assets
        coin.scale(0.3).move_to(circle_a.get_center())
        cat.move_to(circle_b.get_center())
        
        self.play(FadeIn(coin), FadeIn(cat))
        self.play(
            circle_a.animate.set_fill(opacity=0.6),
            circle_b.animate.set_fill(opacity=0.6),
            run_time=1.0
        )
        self.play(
            circle_a.animate.set_fill(opacity=0.3),
            circle_b.animate.set_fill(opacity=0.3),
            run_time=1.0
        )
        self.play(FadeOut(coin), FadeOut(cat))
        self.wait(2)
