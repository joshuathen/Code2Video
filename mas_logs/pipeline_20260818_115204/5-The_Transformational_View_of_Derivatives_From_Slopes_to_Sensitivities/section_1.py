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
        self.setup_layout("Prerequisite Refresh: The Static Slope", [
            "Slope is defined as rise over run.",
            "Static objects have a constant slope.",
            "Curved ramps have a varying steepness."
        ])
        
        # Load asset
        ramp_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ramp.svg")
        
        # Setup Axes - Fixed per VideoCritic
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"include_tip": True}).scale(0.6)
        self.place_in_area(axes, "D2", "F6", scale_factor=0.6)
        self.add(axes)
        
        # Incorporating ramp asset as part of the background or decoration
        self.place_at_grid(ramp_icon, "C2", scale_factor=0.5)
        
        line = Line(axes.c2p(1, 1), axes.c2p(5, 5), color=WHITE)
        point_a = Dot(axes.c2p(1, 1), color="#FFFF00")
        point_b = Dot(axes.c2p(5, 5), color="#00FFFF")
        label_a = Text("A", font_size=20, color="#FFFF00").next_to(point_a, DOWN)
        label_b = Text("B", font_size=20, color="#00FFFF").next_to(point_b, UP)
        
        # Fixed formula position per VideoCritic
        formula = MathTex(r"m = \frac{\text{rise}}{\text{run}}", color=WHITE).scale(0.8)
        self.place_at_grid(formula, "B2", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(axes), Create(line), FadeIn(ramp_icon))
        self.play(FadeIn(point_a), FadeIn(point_b), Write(label_a), Write(label_b))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(BLUE))
        self.play(line.animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(BLUE))
        self.play(Write(formula))
        self.wait(2)
