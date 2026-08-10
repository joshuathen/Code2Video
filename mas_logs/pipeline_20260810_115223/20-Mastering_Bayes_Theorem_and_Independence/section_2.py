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
        self.setup_layout("The Concept of Independence", [
            "- Independent events do not influence each other's outcome.",
            "- Mathematically, P(A|B) equals P(A) for independent events.",
            "- Visualizing independence: two bubbles operating completely separately."
        ])
        
        # Assets
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        
        # Setup Circles
        circle_a = Circle(color="#00FFFF", fill_opacity=0.3)
        circle_b = Circle(color="#00FFFF", fill_opacity=0.3)
        label_a = MathTex("A")
        label_b = MathTex("B")
        
        # Positioning based on feedback (Fix 25)
        self.place_in_area(circle_a, 'B2', 'C3', scale_factor=0.6)
        label_a.next_to(circle_a, UP)
        self.place_in_area(circle_b, 'B4', 'C5', scale_factor=0.6)
        label_b.next_to(circle_b, UP)
        
        venn = VGroup(circle_a, circle_b, label_a, label_b)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(venn))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        math_eq = MathTex("P(A|B)", "=", "P(A)").scale(1.2)
        # Position fixed based on feedback (Fix 24, 26)
        self.place_at_grid(math_eq, 'D4', scale_factor=0.8)
        self.play(Write(math_eq))
        self.lecture[1].set_color(YELLOW)
        
        # Flashing equality sign (Storyboard requirement)
        equality_sign = math_eq[1]
        self.play(Indicate(equality_sign, color="#FFFF00"))

        # === Animation for Lecture Line 3 ===
        coin1 = SVGMobject(coin_path).set_color("#00FF00")
        coin2 = SVGMobject(coin_path).set_color("#00FF00")
        
        self.place_at_grid(coin1, 'F2', scale_factor=0.4)
        self.place_at_grid(coin2, 'F5', scale_factor=0.4)
        
        self.play(FadeIn(coin1), FadeIn(coin2))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
