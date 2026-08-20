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
            "Growth starts with a base seed.",
            "The exponent is the growth engine.",
            "Base two branches each step.",
            "Repeated doubling creates exponential power.",
            "Three steps result in eight."
        ]
        self.setup_layout("The Foundation: Exponential Growth", lecture_lines)
        
        # Setup visual elements
        func = MathTex("f(x) = b^x", color=WHITE)
        seed = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg")
        branch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/branch.svg")
        
        # Position Assets
        self.place_in_area(func, 'A4', 'B6', scale_factor=1.0)
        self.place_at_grid(seed, 'A4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(func), FadeIn(seed))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(func[0][2]), run_time=1.5) # Highlight 'b'
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        curve = Axes(x_range=[0, 3], y_range=[0, 8]).scale(0.4)
        self.place_at_grid(curve, 'D4', scale_factor=0.8)
        self.place_at_grid(branch, 'D6', scale_factor=0.5)
        self.play(Create(curve), FadeIn(branch))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        dot = Dot(color=RED).move_to(curve.c2p(0, 1))
        self.add(dot)
        self.play(MoveAlongPath(dot, curve.plot(lambda x: 2**x)), run_time=2)
        self.lecture[3].set_color("#FF69B4")

        # === Animation for Lecture Line 5 ===
        result = MathTex("2^3 = 8", color=WHITE)
        self.place_at_grid(result, 'C5', scale_factor=0.9)
        self.play(FadeIn(result))
        self.lecture[4].set_color("#7FFF00")
        self.wait(1)
