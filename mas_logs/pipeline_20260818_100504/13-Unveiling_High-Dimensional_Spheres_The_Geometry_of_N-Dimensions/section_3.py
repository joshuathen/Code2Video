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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "High-dimensional volume behaves in counter-intuitive ways.",
            "Volume concentrates near the equator as n increases.",
            "Total volume tends toward zero as n grows.",
            "Imagine an orange in 100-dimensional space.",
            "Almost all fruit hides in the outer layer."
        ]
        self.setup_layout("The Paradox of Volume and Surface Area", lecture_lines)
        
        # Setup animation elements
        orange = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        vol_formula = MathTex("V_n(R) = \\frac{\\pi^{n/2}}{\\Gamma(n/2 + 1)}R^n", color="#FF5733")
        graph = Axes(x_range=[0, 10, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False}).scale(0.5)
        curve = graph.plot(lambda x: 1 / (x + 1), color="#33FF57")
        paradox_label = Text("Volume Paradox", font_size=30, color="#3357FF")
        
        elements = VGroup(orange, vol_formula, graph, curve, paradox_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.place_at_grid(orange, 'B3', scale_factor=0.8)
        self.place_at_grid(vol_formula, 'B3', scale_factor=0.7)
        self.play(FadeIn(orange), FadeIn(vol_formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.place_in_area(graph, 'D2', 'F4', scale_factor=0.75)
        self.play(Create(graph), Create(curve))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        self.place_at_grid(paradox_label, 'B5', scale_factor=0.7)
        self.play(FadeIn(paradox_label))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.play(Flash(elements, color="#FFFFFF"))
        self.wait(2)
