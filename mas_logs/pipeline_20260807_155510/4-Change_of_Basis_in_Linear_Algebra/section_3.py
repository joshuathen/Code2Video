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
            "Transition matrix P links different coordinate systems.",
            "P maps new basis back to standard basis.",
            "Standard coordinates equal P times new coordinates.",
            "View vectors as room-centric or claw-centric.",
            "Projection visually maps new components onto standard axes."
        ]
        self.setup_layout("The Change-of-Basis Matrix", lecture_lines)
        
        # Assets
        room_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/room.svg")
        claw_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/claw.svg")
        
        # Matrix Setup
        matrix_p = Matrix([["p_{11}", "p_{12}"], ["p_{21}", "p_{22}"]]).set_color(WHITE)
        self.place_at_grid(matrix_p, 'B4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(matrix_p), FadeIn(self.place_at_grid(room_icon, 'A4', scale_factor=0.5)))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(matrix_p.animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        eq = MathTex(r"[v]_{std} = P [v]_{new}").set_color("#FFFF00")
        self.place_at_grid(eq, 'D4', scale_factor=0.9)
        self.play(Write(eq))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        # Visual scaling of P
        self.play(
            matrix_p.animate.set_color("#FF00FF").scale(1.2),
            eq.animate.set_color("#FF00FF"),
            FadeIn(self.place_at_grid(claw_icon, 'F4', scale_factor=0.5))
        )

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        # Visual of columns
        col1 = matrix_p.get_columns()[0].copy().set_color("#00FF00")
        self.play(col1.animate.shift(RIGHT * 1.5))
        self.wait(1)
