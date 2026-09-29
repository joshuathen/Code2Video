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
        lecture_lines = [
            "Differentiation and integration are connected.",
            "They are two sides of one coin.",
            "The Fundamental Theorem links them perfectly.",
            "Area under velocity recovers position.",
            "This solves complex physics problems easily."
        ]
        self.setup_layout("The Fundamental Theorem of Calculus", lecture_lines)
        
        # Elements
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        icon2 = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        graph_box = Rectangle(width=4, height=3, color=WHITE)
        arrow = DoubleArrow(start=self.grid["C3"], end=self.grid["C5"], color=YELLOW)
        eqn = MathTex(r"\int_a^b f'(x) dx = f(b) - f(a)", color="#00FFFF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_at_grid(graph_box, "B3", 1.0)
        self.place_at_grid(icon1, "B2", 0.5)
        self.play(Create(graph_box), FadeIn(icon1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(arrow, "C3", 1.0)
        self.play(Create(arrow))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeOut(arrow), FadeOut(graph_box), FadeOut(icon1))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(PINK)
        # Simplified area/velocity visual
        axes = Axes(x_range=[0, 3], y_range=[0, 3], axis_config={"include_tip": False}).scale(0.5)
        self.place_at_grid(axes, "C3", 0.8)
        self.place_at_grid(icon2, "E5", 0.5)
        self.play(Create(axes), FadeIn(icon2))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.place_at_grid(eqn, "C3", 0.8)
        self.play(Write(eqn))
        self.wait(2)
