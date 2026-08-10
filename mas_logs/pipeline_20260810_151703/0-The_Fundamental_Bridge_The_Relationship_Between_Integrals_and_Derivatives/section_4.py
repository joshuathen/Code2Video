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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Practical Application & Summary", [
            "Know the rate of change to reconstruct accumulation.",
            "The anti-derivative recovers the original total amount.",
            "Calculus provides a path from rates to totals.",
            "Explorer Alex tracks speed to find total distance.",
            "This completes our Fundamental Theorem of Calculus summary."
        ])
        
        # Elements
        rate_eq = MathTex(r"f(x) = \frac{d}{dx}F(x)", color=YELLOW)
        total_eq = MathTex(r"\int_{a}^{b} f(x) dx = F(b) - F(a)", color=BLUE)
        
        # Assets
        explorer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/explorer.svg")
        odometer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix 38: use place_in_area
        self.place_in_area(rate_eq, 'B2', 'B5', scale_factor=0.7)
        self.place_at_grid(explorer_icon, 'B1', scale_factor=0.5)
        self.play(Write(rate_eq), FadeIn(explorer_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        # Fix 39: use place_in_area
        self.place_in_area(total_eq, 'D2', 'D5', scale_factor=0.7)
        self.place_at_grid(odometer_icon, 'D1', scale_factor=0.5)
        self.play(Write(total_eq), FadeIn(odometer_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        path_arrow = Arrow(rate_eq.get_bottom(), total_eq.get_top(), color=GREEN)
        self.play(GrowArrow(path_arrow))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        alex_label = Text("Explorer Alex", font_size=24, color=ORANGE)
        # Fix 40: use place_at_grid with E2
        self.place_at_grid(alex_label, 'E2', scale_factor=0.8)
        self.play(FadeIn(alex_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(FadeOut(rate_eq), FadeOut(total_eq), FadeOut(path_arrow), 
                  FadeOut(alex_label), FadeOut(explorer_icon), FadeOut(odometer_icon))
        self.wait(1)
