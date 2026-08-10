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
        lecture_lines = [
            "Energy cascades from large to small eddies.",
            "Richardson process breaks swirls down.",
            "Kolmogorov constant C is about 1.5.",
            "Constant links energy dissipation to spectrum.",
            "Whirlpools demonstrate universal scale breaking."
        ]
        self.setup_layout("The Kolmogorov Cascade & Universal Constants", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Energy cascades from large to small eddies. #00FFFF color
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        large_eddies = VGroup(*[Circle(radius=0.5, color="#00FFFF") for _ in range(2)]).arrange(RIGHT)
        small_eddies = VGroup(*[Circle(radius=0.2, color="#00FFFF") for _ in range(3)]).arrange(RIGHT)
        self.place_at_grid(large_eddies, 'A4', scale_factor=0.8)
        self.place_at_grid(small_eddies, 'C4', scale_factor=0.8)
        self.play(FadeIn(large_eddies), FadeIn(small_eddies))

        # === Animation for Lecture Line 2 ===
        # Richardson process breaks swirls down. #00FF00 color
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        arrow = Arrow(start=UP, end=DOWN, color="#00FF00")
        self.place_at_grid(arrow, 'B4', scale_factor=0.7)
        self.play(GrowArrow(arrow))

        # === Animation for Lecture Line 3 ===
        # Kolmogorov constant C is about 1.5. #FFFFFF color
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        constant_tex = MathTex(r"C", r"\approx", r"1.5", color="#FFFFFF")
        self.place_at_grid(constant_tex, 'D4', scale_factor=0.7)
        self.play(Write(constant_tex))

        # === Animation for Lecture Line 4 ===
        # Constant links energy dissipation to spectrum. #FF00FF color
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        epsilon_tex = MathTex(r"\epsilon", color="#FF00FF")
        self.place_at_grid(epsilon_tex, 'E4', scale_factor=0.7)
        self.play(Indicate(epsilon_tex))

        # === Animation for Lecture Line 5 ===
        # Whirlpools demonstrate universal scale breaking. #FFFFFF color
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        whirlpool = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/whirlpool.svg", color="#FFFFFF")
        self.place_in_area(whirlpool, 'A3', 'F6', scale_factor=0.6)
        self.play(FadeIn(whirlpool))
        self.wait(2)
