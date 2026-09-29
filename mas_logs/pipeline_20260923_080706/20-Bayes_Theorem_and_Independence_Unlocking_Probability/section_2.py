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
            "Independent events do not influence each other.",
            "Outcome A doesn't change probability of B.",
            "Like flipping coins and rolling dice."
        ])

        # Objects
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", color=WHITE)
        sample_space = Rectangle(width=3, height=3, color=WHITE)
        event_b = Circle(radius=0.8, color=BLUE, fill_opacity=0.3)
        event_a = Circle(radius=0.8, color=RED, fill_opacity=0.3)
        
        # Intersection asset
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        intersection = Intersection(event_a, event_b, color=YELLOW, fill_opacity=0.8)
        intersection.set_stroke(color="#FFFF00", width=4)
        
        # Setup visuals on right side (following feedback)
        self.place_at_grid(formula, "A3", scale_factor=0.7)
        self.place_in_area(sample_space, "B4", "F6", scale_factor=0.6)
        self.place_in_area(event_a, "C4", "D5", scale_factor=0.5)
        self.place_in_area(event_b, "D5", "E6", scale_factor=0.5)
        
        self.add(sample_space, event_a, event_b)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(FadeIn(intersection))
        
        # Overlay coin icon
        coin_icon.scale(0.3).move_to(intersection.get_center())
        self.play(FadeIn(coin_icon))
        
        # Dimming effect
        self.play(
            event_a.animate.set_fill(opacity=0.1),
            event_b.animate.set_fill(opacity=0.1),
            sample_space.animate.set_stroke(color="#333333")
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)
