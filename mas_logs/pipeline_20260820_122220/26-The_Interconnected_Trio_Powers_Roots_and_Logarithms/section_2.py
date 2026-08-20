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
            "Roots help us find the growth rate.",
            "If growth is 8 over 3 time.",
            "We calculate $\\sqrt[3]{8}$ to find the base.",
            "The base rate here is exactly 2.",
            "Roots reverse the exponential growth process."
        ]
        self.setup_layout("The Root: Finding the Growth Rate", lecture_lines)
        
        # Mobjects
        axes = Axes(x_range=[0, 4], y_range=[0, 10], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: 2**x, color=WHITE)
        point = Dot(color="#FF4500")
        point.move_to(axes.c2p(3, 8))
        
        # Load asset for root icon
        root_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/root.svg")
        root_icon.set_color("#FF4500")
        
        base_text = MathTex("2", color="#FFD700")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        self.place_in_area(axes, "B3", "D5", scale_factor=0.5)
        self.play(Create(axes), Create(curve))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#87CEEB"))
        self.place_at_grid(point, "D4", scale_factor=0.8)
        self.play(FadeIn(point))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        formula = MathTex("\\sqrt[3]{8} = 2", color=WHITE)
        self.place_at_grid(root_icon, "E2", scale_factor=0.5)
        self.place_at_grid(formula, "E3", scale_factor=0.8)
        self.play(FadeIn(root_icon), Write(formula))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        self.place_at_grid(base_text, "E5", scale_factor=1.2)
        self.play(FadeIn(base_text))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        self.play(Indicate(curve))
        self.wait(1)
