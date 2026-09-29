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
        self.setup_layout("Defining the Variables & Prerequisite Laws", [
            "Conservation of energy defines motion.",
            "Balance distance versus acceleration.",
            "Steeper starts gain speed faster."
        ])
        
        # Set up coordinates for points
        point_a = Dot(color=WHITE)
        point_b = Dot(color=WHITE)
        self.place_at_grid(point_a, "B3", scale_factor=0.7)
        self.place_at_grid(point_b, "E4", scale_factor=0.7)
        label_a = Text("A", font_size=24, color=WHITE).next_to(point_a, UP)
        label_b = Text("B", font_size=24, color=WHITE).next_to(point_b, DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(FadeIn(point_a), FadeIn(point_b), Write(label_a), Write(label_b))

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg]
        path_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg", color="#FF4500")
        line = Line(point_a.get_center(), point_b.get_center(), color="#FF4500")
        self.place_at_grid(path_icon, "C3", scale_factor=0.5)
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.play(Create(line), FadeIn(path_icon))

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/gravity.svg]
        g_label = MathTex(r"g", color="#FFD700", font_size=36)
        gravity_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gravity.svg", color="#FFD700")
        self.place_at_grid(g_label, "D3", scale_factor=0.6)
        self.place_at_grid(gravity_icon, "D4", scale_factor=0.5)
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(FadeIn(g_label), FadeIn(gravity_icon))
        self.wait(2)
