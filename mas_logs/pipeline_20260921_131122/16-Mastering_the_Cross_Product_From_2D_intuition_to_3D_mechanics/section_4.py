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
            "Magnitude equals the 3D parallelogram area.",
            "Formula: magnitude of A times B times sin theta.",
            "Physics application: Torque equals r cross F."
        ]
        self.setup_layout("Geometric Meaning & Real-World Application", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Parallelogram using polygon, wrench icon asset
        parallelogram = Polygon(
            np.array([0, 0, 0]), np.array([1, 0.25, 0]), np.array([1.25, 1, 0]), np.array([0.25, 0.75, 0]),
            color="#FFFFFF"
        )
        wrench = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg", color="#FFFFFF")
        group1 = VGroup(parallelogram, wrench).arrange(RIGHT)
        self.place_in_area(group1, 'A3', 'B5', scale_factor=0.9)
        self.play(Create(group1))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Representing magnitude of A * B * sin(theta) as formula
        formula = MathTex(r"|A| |B| \sin(\theta)", color="#66FF66")
        self.place_at_grid(formula, 'C4', scale_factor=1.0)
        self.play(Write(formula))
        self.play(self.lecture[1].animate.set_color("#66FF66"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Representing Torque = r x F, wrench icon asset
        torque = MathTex(r"\tau = r \times F", color="#FFFF66")
        wrench2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg", color="#FFFF66")
        group2 = VGroup(torque, wrench2).arrange(RIGHT)
        self.place_at_grid(group2, 'E4', scale_factor=1.0)
        self.play(Write(group2))
        self.play(self.lecture[2].animate.set_color("#FFFF66"))
        self.wait(2)
