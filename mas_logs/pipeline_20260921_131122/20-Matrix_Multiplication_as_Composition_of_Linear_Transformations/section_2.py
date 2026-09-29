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
        self.setup_layout("The Concept of Composition", [
            "Sequence two transformations for complex results.",
            "Apply transformation B then apply A.",
            "The combined effect creates composition."
        ])
        
        # Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # === Animation for Lecture Line 1 ===
        f_label = MathTex("f", color="#4169E1")
        g_label = MathTex("g", color="#32CD32")
        self.place_at_grid(f_label, "A3", scale_factor=1.5)
        self.place_at_grid(g_label, "A5", scale_factor=1.5)
        self.place_at_grid(robot, "B4", scale_factor=0.5)
        self.play(Write(f_label), Write(g_label), FadeIn(robot))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        v_label = MathTex("v")
        comp = MathTex("g(f(v))")
        self.place_at_grid(v_label, "C2", scale_factor=1.2)
        self.place_in_area(comp, "C4", "D6", scale_factor=1.2)
        self.play(Write(v_label), Write(comp))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        highlight = SurroundingRectangle(comp, color=WHITE, buff=0.2)
        self.place_at_grid(camera, "E4", scale_factor=0.5)
        self.play(Create(highlight), FadeIn(camera))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
