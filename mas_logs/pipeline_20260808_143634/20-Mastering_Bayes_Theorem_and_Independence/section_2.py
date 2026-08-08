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
            "Independence means B provides no info about A.",
            "Mathematically, P(A|B) equals P(A).",
            "Flipping coins and dice are independent events."
        ])
        
        # Mobjects for animations
        circle_a = Circle(color=WHITE, radius=0.6)
        label_a = Text("A", color=WHITE).next_to(circle_a, UP)
        event_a = VGroup(circle_a, label_a)
        
        circle_b = Circle(color=WHITE, radius=0.6)
        label_b = Text("B", color=WHITE).next_to(circle_b, UP)
        event_b = VGroup(circle_b, label_b)
        
        math_eq = MathTex(r"P(A \cap B) = P(A)P(B)", color=WHITE)
        prob_eq = MathTex(r"P(A|B) = P(A)", color="#00FF00")
        
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        die = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(event_a, 'B4', scale_factor=0.8)
        self.place_at_grid(event_b, 'B6', scale_factor=0.8)
        self.place_at_grid(coin, 'C4', scale_factor=0.5) # Asset integration for coin
        self.play(Create(event_a), Create(event_b), FadeIn(coin))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(math_eq, 'D3', 'D6', scale_factor=0.7)
        self.place_in_area(prob_eq, 'E3', 'E6', scale_factor=0.7)
        self.play(Write(math_eq), Write(prob_eq))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(die, 'F6', scale_factor=0.5) # Asset integration for die
        self.play(FadeIn(die))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
